"""Corpus acquisition, auditing, splitting, tokenization and batching.

Everything here exists to satisfy one rule from SKILL.md: the data pipeline is
part of the deliverable, and provenance is recorded before a single gradient
step. ``prepare`` writes ``manifest.json`` next to the token files containing
the source URL, checksum, license, access date, split boundaries, duplicate and
sensitive-data scan results, and the tokenizer measurement -- everything the
experiment-plan template asks for under "Data card".

Two corpora are registered. ``tiny-shakespeare`` is one file: the fastest way
to prove the pipeline works. ``gutenberg-mixed-27`` is 27 public-domain books
(18.8M characters) pulled from GITenberg repositories, each one stripped of
Project Gutenberg boilerplate, hashed individually, and recorded with its own
title, author, year and source URL -- because a corpus-level hash cannot answer
"where did this sentence come from", which is the first question asked when a
model emits something it should not have.

Split policy: contiguous, never shuffled. A shuffled split puts adjacent,
overlapping context windows on both sides of the train/validation boundary and
leaks. For a multi-document corpus each document is split contiguously and in
the same proportions, then the pieces are concatenated per split: splitting
whole documents between splits would be cleaner for leakage but would make
validation a different set of authors from training, measuring domain transfer
instead of language modelling. Boundaries snap to a blank line so no window
starts mid-paragraph, with a fallback to exact character boundaries for
documents too short to snap -- otherwise a short document is trained on and
never evaluated on. Documents are separated by the tokenizer's EOS id, which
contributes no bytes and is therefore excluded from both sides of bits-per-byte.
"""

from __future__ import annotations

import array
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from tokenizer import BaseTokenizer, build_tokenizer

# --------------------------------------------------------------------------
# Corpus registry. Provenance is data, not a comment.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Corpus:
    name: str
    description: str
    sha256: str
    urls: Sequence[str]
    member: str | None  # path inside the archive, if the URL is an archive
    text_license: str
    file_license: str
    restrictions: str
    access_date: str


TINY_SHAKESPEARE = Corpus(
    name="tiny-shakespeare",
    description=(
        "~1.1 MB of concatenated Shakespeare play text, the standard character-level "
        "language-modelling toy corpus introduced by karpathy/char-rnn."
    ),
    sha256="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
    urls=(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "https://codeload.github.com/karpathy/char-rnn/tar.gz/refs/heads/master",
    ),
    member="char-rnn-master/data/tinyshakespeare/input.txt",
    text_license="Public domain (William Shakespeare, d. 1616); no copyright subsists in the text.",
    file_license="Distributed in karpathy/char-rnn, which is MIT licensed.",
    restrictions=(
        "None known for research or educational use. Early modern English only; "
        "not representative of contemporary text, and not suitable for evaluating "
        "general language ability."
    ),
    access_date="2026-10-17",
)

@dataclass(frozen=True)
class Document:
    """One source document, with its own provenance.

    A multi-document corpus that records only a corpus-level hash cannot answer
    "where did this sentence come from", which is the first question asked when
    a model emits something it should not have.
    """

    doc_id: str
    title: str
    author: str
    year: str
    repo: str
    member: str  # path inside the repo tarball
    train_only: bool = False
    """Contribute to the training split only, never to val or test.

    This exists so a corpus can be *grown* without moving the evaluation
    target. Adding documents the ordinary way changes the val and test splits
    too, and then a bits-per-byte figure from the larger corpus cannot be
    compared with one from the smaller -- the denominator is different text.
    Marking the additions train-only keeps val and test byte-identical, so the
    only thing that changed is how much the model read.
    """

    eval_only: bool = False
    """Contribute to val and test only, never to the training split.

    The mirror of ``train_only``, and the other half of a clean data-scaling
    experiment. ``train_only`` answers "does more training text help on a fixed
    evaluation"; it cannot answer "does more training text help" at all,
    because its evaluation never covers the new material -- and the measured
    answer to the first question came out negative precisely because of that
    gap. To ask the second, the additions have to appear in val and test too,
    and then the *control* run needs a corpus with the same evaluation splits
    and without the extra training text. That is this flag: the document is
    split in the usual proportions, its val and test pieces are kept, and its
    training piece is dropped.

    A document cannot be both; ``__post_init__`` rejects it.
    """

    def __post_init__(self) -> None:
        if self.train_only and self.eval_only:
            raise ValueError(
                f"document {self.doc_id!r} is marked both train_only and eval_only; "
                "a document contributes to the training split, to the evaluation "
                "splits, or to both, but 'neither' is not a corpus worth building"
            )


@dataclass(frozen=True)
class MultiDocCorpus:
    name: str
    description: str
    sha256: str  # of the assembled, boilerplate-stripped, normalised corpus
    documents: Sequence[Document]
    text_license: str
    file_license: str
    restrictions: str
    access_date: str


def _gb(pg: int, slug: str, title: str, author: str, year: str, train_only: bool = False,
        eval_only: bool = False, member_file: str | None = None) -> Document:
    """GITenberg mirrors Project Gutenberg as one git repository per book.

    Most repos carry ``<id>-0.txt`` (UTF-8) and ``<id>.txt`` (legacy encoding),
    and the UTF-8 one is the default here. Not every repo has it: a sizeable
    minority -- including most of the drama and verse -- ships only
    ``<id>.txt``. Those are usable when, and only when, the bytes decode as
    strict UTF-8, which `resolve-broad.py` checks before emitting
    ``member_file``. Guessing instead, by taking the largest ``.txt`` in the
    archive, would silently mix Latin-1 into the corpus.

    The repos were last pushed in 2018, so a tarball of ``master`` is a stable
    snapshot -- which is why this is reproducible and a live gutenberg.org
    fetch would not be.
    """
    return Document(
        doc_id=f"pg{pg}",
        title=title,
        author=author,
        year=year,
        repo=f"GITenberg/{slug}_{pg}",
        member=f"{slug}_{pg}-master/{member_file or f'{pg}-0.txt'}",
        train_only=train_only,
        eval_only=eval_only,
    )


# 27 titles, resolved against the GitHub API on 2026-10-18. Chosen for breadth
# of author, period and register rather than size: novels, short stories,
# verse, folk tales, autobiography and scientific prose, 1605-1915.
GUTENBERG_DOCS = (
    _gb(2701, "Moby-Dick--Or-The-Whale", "Moby Dick", "Herman Melville", "1851"),
    _gb(1342, "Pride-and-Prejudice", "Pride and Prejudice", "Jane Austen", "1813"),
    _gb(158, "Emma", "Emma", "Jane Austen", "1815"),
    _gb(161, "Sense-and-Sensibility", "Sense and Sensibility", "Jane Austen", "1811"),
    _gb(84, "Frankenstein", "Frankenstein", "Mary Shelley", "1818"),
    _gb(345, "Dracula", "Dracula", "Bram Stoker", "1897"),
    _gb(1260, "Jane-Eyre", "Jane Eyre", "Charlotte Bronte", "1847"),
    _gb(768, "Wuthering-Heights", "Wuthering Heights", "Emily Bronte", "1847"),
    _gb(1400, "Great-Expectations", "Great Expectations", "Charles Dickens", "1861"),
    _gb(98, "A-Tale-of-Two-Cities", "A Tale of Two Cities", "Charles Dickens", "1859"),
    _gb(11, "Alice-s-Adventures-in-Wonderland", "Alice's Adventures in Wonderland", "Lewis Carroll", "1865"),
    _gb(76, "Adventures-of-Huckleberry-Finn", "Adventures of Huckleberry Finn", "Mark Twain", "1884"),
    _gb(74, "The-Adventures-of-Tom-Sawyer", "The Adventures of Tom Sawyer", "Mark Twain", "1876"),
    _gb(120, "Treasure-Island", "Treasure Island", "Robert Louis Stevenson", "1883"),
    _gb(174, "The-Picture-of-Dorian-Gray", "The Picture of Dorian Gray", "Oscar Wilde", "1890"),
    _gb(35, "The-Time-Machine", "The Time Machine", "H. G. Wells", "1895"),
    _gb(36, "The-War-of-the-Worlds", "The War of the Worlds", "H. G. Wells", "1898"),
    _gb(1661, "The-Adventures-of-Sherlock-Holmes", "The Adventures of Sherlock Holmes", "Arthur Conan Doyle", "1892"),
    _gb(219, "Heart-of-Darkness", "Heart of Darkness", "Joseph Conrad", "1899"),
    _gb(514, "Little-Women", "Little Women", "Louisa May Alcott", "1868"),
    _gb(45, "Anne-of-Green-Gables", "Anne of Green Gables", "L. M. Montgomery", "1908"),
    _gb(23, "Narrative-of-the-Life-of-Frederick-Douglass-an-American-Slave",
        "Narrative of the Life of Frederick Douglass", "Frederick Douglass", "1845"),
    _gb(2591, "Grimms-Fairy-Tales", "Grimms' Fairy Tales", "Jacob and Wilhelm Grimm", "1812"),
    _gb(1322, "Leaves-of-Grass", "Leaves of Grass", "Walt Whitman", "1855"),
    _gb(1228, "On-the-Origin-of-Species-By-Means-of-Natural-Selection--13-Or-the-Preservation-of-Favoured-Rac_",
        "On the Origin of Species", "Charles Darwin", "1859"),
    _gb(2554, "Crime-and-Punishment", "Crime and Punishment", "Fyodor Dostoyevsky (tr. Garnett)", "1866"),
    _gb(996, "Don-Quixote", "Don Quixote", "Miguel de Cervantes (tr. Ormsby)", "1605"),
)

GUTENBERG_MIXED = MultiDocCorpus(
    name="gutenberg-mixed-27",
    description=(
        "27 public-domain books from Project Gutenberg, mirrored by GITenberg as one git "
        "repository per title. Project Gutenberg's header, footer and licence boilerplate "
        "are stripped before use; what remains is the body text of each work."
    ),
    sha256="1960eb6375c67d9e0f0b4fb96f0ab0724def0d74c547b3baba83faff851dd236",
    documents=GUTENBERG_DOCS,
    text_license=(
        "Public domain in the United States: every title was published before 1929 and its "
        "author died more than 70 years ago, so no copyright subsists in the text itself."
    ),
    file_license=(
        "The Project Gutenberg files carry PG's own licence, which governs the PG header, "
        "footer and trademark rather than the work. That boilerplate is stripped here and "
        "never trained on, which is the condition PG states for using the text as an "
        "ordinary public-domain work. GITenberg's repository metadata is CC0/MIT."
    ),
    restrictions=(
        "Literary English of 1605-1915, mostly British and American, two works in "
        "translation. It is not contemporary text, it is not balanced across dialects or "
        "demographics, and several works contain racial slurs, antisemitism and other "
        "period prejudice that a model trained on them will reproduce."
    ),
    access_date="2026-10-18",
)

# 44 further titles, resolved against the GitHub search API and verified by
# downloading each tarball on 2026-10-19 -- see
# `experiments/2026-10-18-gutenberg-mixed/resolve-gitenberg.py`, which prints
# exactly these lines. 21 further candidates were rejected because their
# GITenberg repository has no canonical `<id>-0.txt` UTF-8 member; taking the
# largest `.txt` instead would have silently mixed encodings into the corpus.
#
# Every one is `train_only`: they enlarge the training split and leave
# validation and test byte-identical to `gutenberg-mixed-27`, which is the
# only way a bits-per-byte number from this corpus can be compared with one
# from that corpus.
GUTENBERG_EXTRA_DOCS = (
    _gb(2600, "War-and-Peace", "War and Peace", "Leo Tolstoy (tr. Maude)", "1869", train_only=True),
    _gb(1399, "Anna-Karenina", "Anna Karenina", "Leo Tolstoy (tr. Garnett)", "1878", train_only=True),
    _gb(766, "David-Copperfield", "David Copperfield", "Charles Dickens", "1850", train_only=True),
    _gb(580, "The-Pickwick-Papers", "The Pickwick Papers", "Charles Dickens", "1837", train_only=True),
    _gb(963, "Little-Dorrit", "Little Dorrit", "Charles Dickens", "1857", train_only=True),
    _gb(883, "Our-Mutual-Friend", "Our Mutual Friend", "Charles Dickens", "1865", train_only=True),
    _gb(786, "Hard-Times", "Hard Times", "Charles Dickens", "1854", train_only=True),
    _gb(46, "A-Christmas-Carol", "A Christmas Carol", "Charles Dickens", "1843", train_only=True),
    _gb(121, "Northanger-Abbey", "Northanger Abbey", "Jane Austen", "1817", train_only=True),
    _gb(141, "Mansfield-Park", "Mansfield Park", "Jane Austen", "1814", train_only=True),
    _gb(2852, "The-Hound-of-the-Baskervilles", "The Hound of the Baskervilles", "Arthur Conan Doyle", "1902", train_only=True),
    _gb(244, "A-Study-in-Scarlet", "A Study in Scarlet", "Arthur Conan Doyle", "1887", train_only=True),
    _gb(834, "The-Memoirs-of-Sherlock-Holmes", "The Memoirs of Sherlock Holmes", "Arthur Conan Doyle", "1894", train_only=True),
    _gb(108, "The-Return-of-Sherlock-Holmes", "The Return of Sherlock Holmes", "Arthur Conan Doyle", "1905", train_only=True),
    _gb(1184, "The-Count-of-Monte-Cristo", "The Count of Monte Cristo", "Alexandre Dumas (tr.)", "1844", train_only=True),
    _gb(1257, "The-Three-Musketeers", "The Three Musketeers", "Alexandre Dumas (tr.)", "1844", train_only=True),
    _gb(2413, "Madame-Bovary", "Madame Bovary", "Gustave Flaubert (tr. Marx-Aveling)", "1856", train_only=True),
    _gb(135, "Les-Mis-rables", "Les Miserables", "Victor Hugo (tr. Hapgood)", "1862", train_only=True),
    _gb(6130, "The-Iliad", "The Iliad", "Homer (tr. Pope)", "-750", train_only=True),
    _gb(4300, "Ulysses", "Ulysses", "James Joyce", "1922", train_only=True),
    _gb(4217, "A-Portrait-of-the-Artist-as-a-Young-Man", "A Portrait of the Artist as a Young Man", "James Joyce", "1916", train_only=True),
    _gb(2814, "Dubliners", "Dubliners", "James Joyce", "1914", train_only=True),
    _gb(160, "The-Awakening-and-Selected-Short-Stories", "The Awakening", "Kate Chopin", "1899", train_only=True),
    _gb(205, "Walden-and-On-The-Duty-Of-Civil-Disobedience", "Walden", "Henry David Thoreau", "1854", train_only=True),
    _gb(215, "The-Call-of-the-Wild", "The Call of the Wild", "Jack London", "1903", train_only=True),
    _gb(521, "The-Life-and-Adventures-of-Robinson-Crusoe", "Robinson Crusoe", "Daniel Defoe", "1719", train_only=True),
    _gb(829, "Gulliver-s-Travels", "Gulliver's Travels", "Jonathan Swift", "1726", train_only=True),
    _gb(209, "The-Turn-of-the-Screw", "The Turn of the Screw", "Henry James", "1898", train_only=True),
    _gb(2641, "A-Room-with-a-View", "A Room with a View", "E. M. Forster", "1908", train_only=True),
    _gb(113, "The-Secret-Garden", "The Secret Garden", "Frances Hodgson Burnett", "1911", train_only=True),
    _gb(271, "Black-Beauty", "Black Beauty", "Anna Sewell", "1877", train_only=True),
    _gb(236, "The-Jungle-Book", "The Jungle Book", "Rudyard Kipling", "1894", train_only=True),
    _gb(25344, "The-Scarlet-Letter", "The Scarlet Letter", "Nathaniel Hawthorne", "1850", train_only=True),
    _gb(1998, "Thus-Spake-Zarathustra--A-Book-for-All-and-None", "Thus Spake Zarathustra", "Friedrich Nietzsche (tr. Common)", "1883", train_only=True),
    _gb(6593, "History-of-Tom-Jones-a-Foundling", "The History of Tom Jones", "Henry Fielding", "1749", train_only=True),
    _gb(28054, "The-Brothers-Karamazov", "The Brothers Karamazov", "Fyodor Dostoyevsky (tr. Garnett)", "1880", train_only=True),
    _gb(2638, "The-Idiot", "The Idiot", "Fyodor Dostoyevsky (tr. Martin)", "1869", train_only=True),
    _gb(203, "Uncle-Tom-s-Cabin", "Uncle Tom's Cabin", "Harriet Beecher Stowe", "1852", train_only=True),
    _gb(16, "Peter-Pan", "Peter Pan", "J. M. Barrie", "1911", train_only=True),
    _gb(289, "The-Wind-in-the-Willows", "The Wind in the Willows", "Kenneth Grahame", "1908", train_only=True),
    _gb(3176, "The-Innocents-Abroad", "The Innocents Abroad", "Mark Twain", "1869", train_only=True),
    _gb(1837, "The-Prince-and-the-Pauper", "The Prince and the Pauper", "Mark Twain", "1881", train_only=True),
    _gb(86, "A-Connecticut-Yankee-in-King-Arthur-s-Court", "A Connecticut Yankee in King Arthur's Court", "Mark Twain", "1889", train_only=True),
    _gb(1952, "The-Yellow-Wallpaper", "The Yellow Wallpaper", "Charlotte Perkins Gilman", "1892", train_only=True),
)

GUTENBERG_EXPANDED = MultiDocCorpus(
    name="gutenberg-mixed-71",
    description=(
        "The 27-book corpus plus 44 further public-domain titles that feed the training "
        "split only. Validation and test are byte-identical to `gutenberg-mixed-27`, so "
        "the two corpora are directly comparable and the only variable is how much text "
        "the model saw."
    ),
    sha256="52290d350e136a6d6e8b24e778e7489183b76389828b1bd8e16898e71d381368",
    documents=GUTENBERG_DOCS + GUTENBERG_EXTRA_DOCS,
    text_license=GUTENBERG_MIXED.text_license,
    file_license=GUTENBERG_MIXED.file_license,
    restrictions=(
        "Literary English of 1320-1922, mostly British and American, with ten works in "
        "translation. It is not contemporary text, it is not balanced across dialects or "
        "demographics, and several works contain racial slurs, antisemitism and other "
        "period prejudice that a model trained on them will reproduce. The additions "
        "widen the period and the register but not the demographics."
    ),
    access_date="2026-10-19",
)

# The clean restatement of the data-scaling question.
#
# `gutenberg-mixed-71` holds its evaluation splits fixed by marking the 44
# additions train-only, which answers "does more training text help on the
# original 27 books' val/test". Measured answer: no, it costs 0.53% test bpb,
# because a 27-book model finds the additions 6.93% harder than its own test
# split -- they are a different distribution, so the extra text competes for
# capacity with the material being scored.
#
# That leaves the general question unanswered, and these two corpora ask it.
# Both carry the same 71 documents and therefore produce the same assembled
# text and the same corpus hash; they differ only in which documents feed the
# training split:
#
#   gutenberg-mixed-71-full  all 71 train, all 71 evaluate
#   gutenberg-mixed-71-ctrl  27 train,     all 71 evaluate
#
# Prepared with a shared vocabulary (`--tokenizer-from`), their val.bin and
# test.bin are byte-identical, so a model from each can be scored on exactly
# the same bytes and the only variable is how much text was trained on.
GUTENBERG_71_FULL_DOCS = tuple(
    replace(d, train_only=False, eval_only=False) for d in GUTENBERG_DOCS + GUTENBERG_EXTRA_DOCS
)

GUTENBERG_71_CTRL_DOCS = tuple(
    replace(d, train_only=False, eval_only=d.train_only) for d in GUTENBERG_DOCS + GUTENBERG_EXTRA_DOCS
)

GUTENBERG_71_FULL = MultiDocCorpus(
    name="gutenberg-mixed-71-full",
    description=(
        "All 71 public-domain titles, every one of them feeding train, val and test. "
        "The treatment arm of the clean data-scaling experiment: unlike "
        "`gutenberg-mixed-71`, the evaluation splits cover the added books, so a win "
        "here is a win on the distribution that was actually trained on."
    ),
    sha256=GUTENBERG_EXPANDED.sha256,
    documents=GUTENBERG_71_FULL_DOCS,
    text_license=GUTENBERG_MIXED.text_license,
    file_license=GUTENBERG_MIXED.file_license,
    restrictions=GUTENBERG_EXPANDED.restrictions,
    access_date="2026-10-19",
)

GUTENBERG_71_CTRL = MultiDocCorpus(
    name="gutenberg-mixed-71-ctrl",
    description=(
        "The control arm: the same 71 titles and the same val and test splits as "
        "`gutenberg-mixed-71-full`, but only the original 27 books contribute to "
        "training. Prepare it with --tokenizer-from pointed at the full arm so the "
        "two share a vocabulary and their evaluation splits are byte-identical."
    ),
    sha256=GUTENBERG_EXPANDED.sha256,
    documents=GUTENBERG_71_CTRL_DOCS,
    text_license=GUTENBERG_MIXED.text_license,
    file_license=GUTENBERG_MIXED.file_license,
    restrictions=GUTENBERG_EXPANDED.restrictions,
    access_date="2026-10-19",
)

# 80 further titles: drama, verse and non-fiction, resolved and byte-verified
# on 2026-10-20 by experiments/2026-10-18-gutenberg-mixed/discover-broad.py.
#
# The per-document analysis of the 71-book corpus found that form explains 45%
# of the spread in test bpb -- verse cost +27.8% over narrative prose,
# experimental prose +23.4%, non-fiction +19.3% -- while those three forms were
# only 9.3% of the bytes, and drama was absent entirely. Repeated authors
# scored 5.0% better than one-off authors, which says the hard forms are
# under-trained rather than simply unpredictable.
#
# These additions are sized to test that: enough non-narrative text to bring
# narrative prose from 90.7% of the corpus down to 67.8%, spread roughly evenly
# over the three forms so no single one carries the result. Smallest-first
# within each form, which buys the most distinct titles for the byte budget.
# All are train_only, so adding them does not move val or test.
GUTENBERG_BROAD_DOCS = (
    _gb(790, "Lady-Windermere-s-Fan", "Lady Windermere's Fan", "Oscar Wilde", "1892", train_only=True),  # drama
    _gb(854, "A-Woman-of-No-Importance", "A Woman of No Importance", "Oscar Wilde", "1893", train_only=True),  # drama
    _gb(885, "An-Ideal-Husband", "An Ideal Husband", "Oscar Wilde", "1895", train_only=True),  # drama
    _gb(1001, "Divine-Comedy-Longfellow-s-Translation-Hell", "Divine Comedy Longfellow s Translation Hell", "", "", train_only=True, member_file="1001.txt"),  # drama
    _gb(1104, "The-Comedy-of-Errors", "The Comedy of Errors", "", "", train_only=True, member_file="1104.txt"),  # drama
    _gb(1112, "The-Tragedy-of-Romeo-and-Juliet", "The Tragedy of Romeo and Juliet", "", "", train_only=True, member_file="1112.txt"),  # drama
    _gb(1504, "The-Comedy-of-Errors", "The Comedy of Errors", "", "", train_only=True, member_file="1504.txt"),  # drama
    _gb(1513, "Romeo-and-Juliet", "Romeo and Juliet", "William Shakespeare", "1597", train_only=True),  # drama
    _gb(1519, "Much-Ado-about-Nothing", "Much Ado About Nothing", "William Shakespeare", "1600", train_only=True),  # drama
    _gb(1522, "Julius-Caesar", "Julius Caesar", "William Shakespeare", "1599", train_only=True),  # drama
    _gb(1524, "Hamlet-Prince-of-Denmark", "Hamlet", "William Shakespeare", "1603", train_only=True),  # drama
    _gb(1532, "The-Tragedy-of-King-Lear", "King Lear", "William Shakespeare", "1606", train_only=True),  # drama
    _gb(1769, "The-Comedy-of-Errors", "The Comedy of Errors", "", "", train_only=True, member_file="1769.txt"),  # drama
    _gb(2379, "Chastelard-a-Tragedy", "Chastelard a Tragedy", "", "", train_only=True, member_file="2379.txt"),  # drama
    _gb(3325, "Locrine-A-Tragedy", "Locrine A Tragedy", "", "", train_only=True),  # drama
    _gb(5618, "Six-Plays", "Six Plays", "", "", train_only=True),  # drama
    _gb(6043, "The-Spanish-Tragedie", "The Spanish Tragedie", "", "", train_only=True, member_file="6043.txt"),  # drama
    _gb(6332, "Playful-Poems", "Playful Poems", "", "", train_only=True),  # drama
    _gb(7805, "First-Plays", "First Plays", "", "", train_only=True, member_file="7805.txt"),  # drama
    _gb(7986, "Plays-by-Anton-Chekhov-Second-Series", "Plays by Anton Chekhov, Second Series", "Anton Chekhov (tr. Fell)", "1916", train_only=True),  # drama
    _gb(10606, "The-Tragedie-of-Hamlet-Prince-of-DenmarkA-Study-with-the-Text-of-the-Folio-of-1623", "The Tragedie of Hamlet Prince of DenmarkA Study with the Text of the Folio of 1623", "", "", train_only=True, member_file="10606.txt"),  # drama
    _gb(10623, "Plays", "Plays", "", "", train_only=True, member_file="10623.txt"),  # drama
    _gb(10722, "Plays", "Plays", "", "", train_only=True, member_file="10722.txt"),  # drama
    _gb(10847, "The-Maids-Tragedy", "The Maids Tragedy", "", "", train_only=True, member_file="10847.txt"),  # drama
    _gb(10877, "The-Tragedies-of-the-Medici", "The Tragedies of the Medici", "", "", train_only=True, member_file="10877.txt"),  # drama
    _gb(11346, "The-Saint-s-Tragedy", "The Saint s Tragedy", "", "", train_only=True, member_file="11346.txt"),  # drama
    _gb(147, "Common-Sense", "Common Sense", "Thomas Paine", "1776", train_only=True),  # non-fiction
    _gb(226, "Cicero-s-Orations", "Cicero s Orations", "", "", train_only=True, member_file="226.txt"),  # non-fiction
    _gb(774, "Essays-and-Lectures", "Essays and Lectures", "", "", train_only=True),  # non-fiction
    _gb(990, "Theologico-Political-Treatise-Part-2", "Theologico Political Treatise Part 2", "", "", train_only=True, member_file="990.txt"),  # non-fiction
    _gb(991, "Theologico-Political-Treatise-Part-3", "Theologico Political Treatise Part 3", "", "", train_only=True, member_file="991.txt"),  # non-fiction
    _gb(992, "Theologico-Political-Treatise-Part-4", "Theologico Political Treatise Part 4", "", "", train_only=True, member_file="992.txt"),  # non-fiction
    _gb(1315, "Autobiography-and-Selected-Essays", "Autobiography and Selected Essays", "", "", train_only=True, member_file="1315.txt"),  # non-fiction
    _gb(1336, "Shelley--an-essay", "Shelley an essay", "", "", train_only=True, member_file="1336.txt"),  # non-fiction
    _gb(1434, "Essays", "Essays", "", "", train_only=True, member_file="1434.txt"),  # non-fiction
    _gb(1594, "Essays-in-Little", "Essays in Little", "", "", train_only=True, member_file="1594.txt"),  # non-fiction
    _gb(2167, "Miscellaneous-Writings-and-Speeches---Volume-1", "Miscellaneous Writings and Speeches Volume 1", "", "", train_only=True, member_file="2167.txt"),  # non-fiction
    _gb(2173, "Thoughts-on-the-Present-Discontents-and-Speeches", "Thoughts on the Present Discontents and Speeches", "", "", train_only=True, member_file="2173.txt"),  # non-fiction
    _gb(2176, "Seven-Discourses-on-Art", "Seven Discourses on Art", "", "", train_only=True, member_file="2176.txt"),  # non-fiction
    _gb(2428, "An-Essay-on-Man--Moral-Essays-and-Satires", "An Essay on Man Moral Essays and Satires", "", "", train_only=True, member_file="2428.txt"),  # non-fiction
    _gb(2458, "Sermons-on-the-Card-and-Other-Discourses", "Sermons on the Card and Other Discourses", "", "", train_only=True, member_file="2458.txt"),  # non-fiction
    _gb(2615, "Discourses-on-Satire-and-on-Epic-Poetry", "Discourses on Satire and on Epic Poetry", "", "", train_only=True),  # non-fiction
    _gb(2945, "Essays---Second-Series", "Essays Second Series", "", "", train_only=True, member_file="2945.txt"),  # non-fiction
    _gb(3549, "Cowley-s-Essays", "Cowley s Essays", "", "", train_only=True),  # non-fiction
    _gb(4220, "An-Autobiography", "An Autobiography", "", "", train_only=True, member_file="4220.txt"),  # non-fiction
    _gb(6312, "Representative-Men-Seven-Lectures", "Representative Men Seven Lectures", "", "", train_only=True, member_file="6312.txt"),  # non-fiction
    _gb(6969, "The-Orations-of-Lysias", "The Orations of Lysias", "", "", train_only=True, member_file="6969.txt"),  # non-fiction
    _gb(7377, "Five-Lectures-on-Reincarnation", "Five Lectures on Reincarnation", "", "", train_only=True, member_file="7377.txt"),  # non-fiction
    _gb(9061, "The-Public-Orations-of-Demosthenes-volume-2", "The Public Orations of Demosthenes volume 2", "", "", train_only=True, member_file="9061.txt"),  # non-fiction
    _gb(9304, "Initiation-into-Philosophy", "Initiation into Philosophy", "", "", train_only=True, member_file="9304.txt"),  # non-fiction
    _gb(10427, "Scientific-Essays-and-Lectures", "Scientific Essays and Lectures", "", "", train_only=True, member_file="10427.txt"),  # non-fiction
    _gb(10956, "Indian-speeches--1907-1909-", "Indian speeches 1907 1909", "", "", train_only=True, member_file="10956.txt"),  # non-fiction
    _gb(11261, "Cyclopedia-of-Philosophy", "Cyclopedia of Philosophy", "", "", train_only=True, member_file="11261.txt"),  # non-fiction
    _gb(228, "The-Aeneid", "The Aeneid", "Virgil (tr. Dryden)", "-19", train_only=True),  # verse
    _gb(413, "Ballads", "Ballads", "", "", train_only=True),  # verse
    _gb(454, "The-Breitmann-Ballads", "The Breitmann Ballads", "", "", train_only=True, member_file="454.txt"),  # verse
    _gb(995, "Ballads-of-a-Bohemian", "Ballads of a Bohemian", "", "", train_only=True, member_file="995.txt"),  # verse
    _gb(1041, "Shakespeare-s-Sonnets", "Shakespeare s Sonnets", "", "", train_only=True, member_file="1041.txt"),  # verse
    _gb(1105, "The-Shakespearian-Sonnets", "The Shakespearian Sonnets", "", "", train_only=True, member_file="1105.txt"),  # verse
    _gb(1469, "Poems", "Poems", "", "", train_only=True),  # verse
    _gb(1745, "The-Poetical-Works-of-John-Milton", "The Poetical Works of John Milton", "", "", train_only=True, member_file="1745.txt"),  # verse
    _gb(1934, "Songs-of-Innocence-and-Songs-of-Experience", "Songs of Innocence and of Experience", "William Blake", "1794", train_only=True),  # verse
    _gb(2002, "Sonnets-from-the-Portuguese", "Sonnets from the Portuguese", "", "", train_only=True),  # verse
    _gb(2507, "Complete-Poetical-Works", "Complete Poetical Works", "", "", train_only=True, member_file="2507.txt"),  # verse
    _gb(2732, "Ballads", "Ballads", "", "", train_only=True, member_file="2732.txt"),  # verse
    _gb(2819, "Barrack-Room-Ballads", "Barrack Room Ballads", "", "", train_only=True),  # verse
    _gb(2888, "Yorkshire-Dialect-Poems--1673-1915--and-traditional-poems", "Yorkshire Dialect Poems 1673 1915 and traditional poems", "", "", train_only=True, member_file="2888.txt"),  # verse
    _gb(4207, "Aesthetic-Poetry", "Aesthetic Poetry", "", "", train_only=True, member_file="4207.txt"),  # verse
    _gb(5432, "The-Odes-and-Carmen-Saeculare-of-Horace", "The Odes and Carmen Saeculare of Horace", "", "", train_only=True, member_file="5432.txt"),  # verse
    _gb(6797, "The-Poems-of-Schiller---Suppressed-poems", "The Poems of Schiller Suppressed poems", "", "", train_only=True, member_file="6797.txt"),  # verse
    _gb(9500, "Farm-Ballads", "Farm Ballads", "", "", train_only=True, member_file="9500.txt"),  # verse
    _gb(9574, "Poems-of-Nature-Poems-Subjective-and-Reminiscent-and-Religious-Poems-Complete--13-Volume-II-of_", "Poems of Nature Poems Subjective and Reminiscent and Religious Poems Complete 13 Volume II of", "", "", train_only=True, member_file="9574.txt"),  # verse
    _gb(9646, "Horace--Odes-and-Epodes", "Horace Odes and Epodes", "", "", train_only=True, member_file="9646.txt"),  # verse
    _gb(9652, "Miscellany-of-Poetry1919", "Miscellany of Poetry1919", "", "", train_only=True, member_file="9652.txt"),  # verse
    _gb(9814, "The-Poetical-Works-of-Mark-Akenside", "The Poetical Works of Mark Akenside", "", "", train_only=True, member_file="9814.txt"),  # verse
    _gb(10367, "Poems", "Poems", "", "", train_only=True, member_file="10367.txt"),  # verse
    _gb(10717, "The-Extant-Odes-of-Pindar--13-Translated-with-Introduction-and-Short-Notes-by-Ernest-Myers", "The Extant Odes of Pindar 13 Translated with Introduction and Short Notes by Ernest Myers", "", "", train_only=True, member_file="10717.txt"),  # verse
    _gb(11236, "Old-Ballads", "Old Ballads", "", "", train_only=True, member_file="11236.txt"),  # verse
    _gb(11496, "Poetry", "Poetry", "", "", train_only=True),  # verse
    _gb(21765, "The-Metamorphoses-of-Ovid-Books-I-VII", "Metamorphoses", "Ovid (tr. More)", "8", train_only=True),  # verse
)

# Treatment and control for the corpus-composition experiment. The treatment
# trains on the 71 originals plus the 80 non-narrative additions; the control
# trains on the 71 alone. Both evaluate on exactly the same bytes -- the
# additions are eval_only in the control and train_only in the treatment, so
# neither arm's val or test split contains them -- and sharing a vocabulary via
# --tokenizer-from makes the two directly comparable. The measurement that
# matters happens on the 71 originals, whose form labels are hand-assigned.
# Computed on 2026-10-20 over the assembled 151-document text, 78,947,277
# characters. Both arms assemble the same documents in the same order, so
# they share it; the control verifying against a hash computed from the
# treatment is the check that they really are the same corpus.
GUTENBERG_BROAD_SHA = "0a3c53a3c9a1321fa38ba197d4c58528e7db6f609d5ca0858f58a36c1d295d81"

GUTENBERG_BROAD_DOCS_FULL = tuple(
    replace(d, train_only=False, eval_only=False) for d in GUTENBERG_DOCS + GUTENBERG_EXTRA_DOCS
) + GUTENBERG_BROAD_DOCS

# The control omits the additions rather than marking them eval_only. That
# distinction is the whole experiment and it is easy to get backwards:
# eval_only puts a document *into* val and test while keeping it out of train,
# which is right for the data-scaling arms -- there the added books were the
# thing being evaluated -- and wrong here. Here the additions are training
# mass and must never be evaluated, so the treatment marks them train_only and
# the control drops them. Both arms then evaluate on exactly the original 71
# books. Measured, not asserted: tests/test_broad_corpus.py compares the two
# prepared val.bin and test.bin byte for byte.
GUTENBERG_BROAD_DOCS_CTRL = tuple(
    replace(d, train_only=False, eval_only=False) for d in GUTENBERG_DOCS + GUTENBERG_EXTRA_DOCS
)

GUTENBERG_BROAD_FULL = MultiDocCorpus(
    name="gutenberg-mixed-broad-full",
    description=(
        "The 71-book corpus plus 80 works of drama, verse and non-fiction, "
        "which takes narrative prose from 90.7% of the bytes to 67.8%. The "
        "treatment arm for the question of whether the per-form penalties are a "
        "property of the forms or of how little of them the model ever saw."
    ),
    sha256=GUTENBERG_BROAD_SHA,
    documents=GUTENBERG_BROAD_DOCS_FULL,
    text_license=GUTENBERG_MIXED.text_license,
    file_license=GUTENBERG_MIXED.file_license,
    restrictions=GUTENBERG_EXPANDED.restrictions,
    access_date="2026-10-20",
)

GUTENBERG_BROAD_CTRL = MultiDocCorpus(
    name="gutenberg-mixed-broad-ctrl",
    description=(
        "The control arm: the original 71 books alone. Because the 80 additions "
        "are train_only in `gutenberg-mixed-broad-full`, dropping them here "
        "leaves val and test byte-identical between the two arms while the "
        "training split loses 7.0M tokens of drama, verse and non-fiction. "
        "Prepare it with --tokenizer-from pointed at the full arm so the two "
        "share a vocabulary."
    ),
    sha256=GUTENBERG_EXPANDED.sha256,   # the 71-book text, unchanged
    documents=GUTENBERG_BROAD_DOCS_CTRL,
    text_license=GUTENBERG_MIXED.text_license,
    file_license=GUTENBERG_MIXED.file_license,
    restrictions=GUTENBERG_EXPANDED.restrictions,
    access_date="2026-10-20",
)

# 31 further narrative titles, 20,464,929 bytes -- within 4,257 bytes (0.02%)
# of what GUTENBERG_BROAD_DOCS adds. This is the volume-matched control.
#
# broad-full trains on 37% more tokens than broad-ctrl *and* on a different
# mixture, so a win there could be either cause. Adding the same number of
# bytes of ordinary narrative prose separates them:
#
#   broad-ctrl       18.9M train tokens, 90.7% narrative
#   broad-narrative  ~25.9M train tokens, ~93% narrative   <- volume matched
#   broad-full       25.9M train tokens, 67.8% narrative   <- mixture changed
#
# full vs narrative is composition holding volume fixed; narrative vs ctrl is
# volume holding composition roughly fixed. Fewer, larger titles than the
# broad arm (31 against 80), because matching bytes is what this arm is for.
# train_only throughout, so val and test do not move.
GUTENBERG_NARRATIVE_DOCS = (
    _gb(106, "Jungle-Tales-of-Tarzan", "Jungle Tales of Tarzan", "", "", train_only=True, member_file="106.txt"),
    _gb(831, "Four-Arthurian-Romances", "Four Arthurian Romances", "", "", train_only=True, member_file="831.txt"),
    _gb(864, "The-Master-of-Ballantrae--A-Winter-s-Tale", "The Master of Ballantrae A Winter s Tale", "", "", train_only=True),
    _gb(1630, "Little-Novels", "Little Novels", "", "", train_only=True),
    _gb(1640, "Lilith--A-Romance", "Lilith A Romance", "", "", train_only=True),
    _gb(1678, "An-Historical-Mystery--The-Gondreville-Mystery-", "An Historical Mystery The Gondreville Mystery", "", "", train_only=True),
    _gb(1828, "Chronicles-of-the-Canongate-1st-Series", "Chronicles of the Canongate 1st Series", "", "", train_only=True),
    _gb(1994, "Adventures-Among-Books", "Adventures Among Books", "", "", train_only=True, member_file="1994.txt"),
    _gb(2021, "Nostromo--A-Tale-of-the-Seaboard", "Nostromo A Tale of the Seaboard", "", "", train_only=True),
    _gb(2736, "The-Champdoce-Mystery", "The Champdoce Mystery", "", "", train_only=True),
    _gb(3261, "News-from-Nowhere--Or-An-Epoch-of-Rest--13-Being-Some-Chapters-from-a-Utopian-Romance", "News from Nowhere Or An Epoch of Rest 13 Being Some Chapters from a Utopian Romance", "", "", train_only=True, member_file="3261.txt"),
    _gb(3492, "Homespun-Tales", "Homespun Tales", "", "", train_only=True),
    _gb(4931, "Won-By-the-Sword---a-tale-of-the-Thirty-Years-War", "Won By the Sword a tale of the Thirty Years War", "", "", train_only=True),
    _gb(5139, "Tales", "Tales", "", "", train_only=True),
    _gb(5217, "Tales", "Tales", "", "", train_only=True, member_file="5217.txt"),
    _gb(5308, "The-Paradise-Mystery", "The Paradise Mystery", "", "", train_only=True),
    _gb(6695, "Tales-of-the-Jazz-Age", "Tales of the Jazz Age", "", "", train_only=True, member_file="6695.txt"),
    _gb(8720, "Tales-and-Novels---Volume-02--13-Popular-Tales", "Tales and Novels Volume 02 13 Popular Tales", "", "", train_only=True),
    _gb(8826, "Tales-and-Novels---Volume-01--13-Moral-Tales", "Tales and Novels Volume 01 13 Moral Tales", "", "", train_only=True, member_file="8826.txt"),
    _gb(9370, "Tiverton-Tales", "Tiverton Tales", "", "", train_only=True, member_file="9370.txt"),
    _gb(10148, "The-Merry-Adventures-of-Robin-Hood", "The Merry Adventures of Robin Hood", "", "", train_only=True, member_file="10148.txt"),
    _gb(11703, "The-Swiss-Family-Robinson--or-Adventures-in-a-Desert-Island", "The Swiss Family Robinson or Adventures in a Desert Island", "", "", train_only=True, member_file="11703.txt"),
    _gb(13731, "Romance-Island", "Romance Island", "", "", train_only=True, member_file="13731.txt"),
    _gb(15616, "Hero-Tales", "Hero Tales", "", "", train_only=True),
    _gb(16012, "The-Ned-M-Keown-StoriesWilliam-Carleton-Volume-Three", "The Ned M Keown StoriesWilliam Carleton Volume Three", "", "", train_only=True),
    _gb(16050, "The-Gold-Hunters-AdventuresOr-Life-in-Australia", "The Gold Hunters AdventuresOr Life in Australia", "", "", train_only=True, member_file="16050.txt"),
    _gb(17452, "Adventures-in-Criticism", "Adventures in Criticism", "", "", train_only=True),
    _gb(17642, "Romance", "Romance", "", "", train_only=True),
    _gb(18304, "American-AdventuresA-Second-Trip-Abroad-at-home-", "American AdventuresA Second Trip Abroad at home", "", "", train_only=True, member_file="18304.txt"),
    _gb(19068, "Household-Stories-by-the-Brothers-Grimm", "Household Stories by the Brothers Grimm", "", "", train_only=True, member_file="19068.txt"),
    _gb(20486, "Tiverton-Tales", "Tiverton Tales", "", "", train_only=True, member_file="20486.txt"),
)

GUTENBERG_NARRATIVE = MultiDocCorpus(
    name="gutenberg-mixed-narrative",
    description=(
        "The volume-matched control: the original 71 books plus 20.5 MB of "
        "further narrative prose, the same number of bytes the broad arm adds "
        "as drama, verse and non-fiction. Narrative stays near 93% of the "
        "corpus. Prepare it with --tokenizer-from pointed at the broad arm."
    ),
    # Computed 2026-10-20 over the assembled 102-document text, 79,569,969
    # characters. Its train split is 26,057,637 tokens against the broad arm's
    # 25,952,894 -- 0.40% apart, which is the volume match this arm exists for.
    sha256="f5048aef958705527b85287a3c11f9691d787720cca51a58f3a6dc7500b27f54",
    documents=tuple(
        replace(d, train_only=False, eval_only=False)
        for d in GUTENBERG_DOCS + GUTENBERG_EXTRA_DOCS
    ) + GUTENBERG_NARRATIVE_DOCS,
    text_license=GUTENBERG_MIXED.text_license,
    file_license=GUTENBERG_MIXED.file_license,
    restrictions=GUTENBERG_EXPANDED.restrictions,
    access_date="2026-10-20",
)

# 40 further titles -- 10 each of drama, verse, non-fiction and narrative
# prose -- resolved and byte-verified on 2026-10-21, none of them in any
# training corpus. This is an evaluation corpus only; nothing trains on it.
#
# It exists because the obvious way to ask "is the broad model more general?"
# is contaminated: in `gutenberg-mixed-broad-full` the 80 additions are
# train_only, and a train_only document contributes its *entire* text to the
# training split, so their would-be test pieces were trained on. Scoring the
# broad model there would measure fit, not transfer. These 40 books are new
# to every model in the project, so a broad-vs-ctrl comparison on them is a
# clean distribution-shift probe, per form. Form labels come from the search
# keyword that found each repository (weaker than the hand labels used for
# the original 71; the selection is in heldout-selection.json for scrutiny).
GUTENBERG_HELDOUT_DOCS = (
    _gb(1191, "The-Double-Dealer--A-Comedy", "The Double Dealer A Comedy", "", "", member_file="1191.txt"),  # drama
    _gb(1507, "The-Tragedy-of-Titus-Andronicus", "The Tragedy of Titus Andronicus", "", "", member_file="1507.txt"),  # drama
    _gb(1535, "The-Tragedy-of-Coriolanus", "The Tragedy of Coriolanus", "", "", member_file="1535.txt"),  # drama
    _gb(1755, "Ivanoff--A-Play", "Ivanoff A Play", "", "", member_file="1755.txt"),  # drama
    _gb(2239, "The-Comedy-of-Errors", "The Comedy of Errors", "", "", member_file="2239.txt"),  # drama
    _gb(3068, "Washington-Square-Plays", "Washington Square Plays", "", "", member_file="3068.txt"),  # drama
    _gb(3142, "Plays-and-Puritans", "Plays and Puritans", "", ""),  # drama
    _gb(4255, "A-Yorkshire-Tragedy", "A Yorkshire Tragedy", "", "", member_file="4255.txt"),  # drama
    _gb(5060, "Six-Short-Plays", "Six Short Plays", "", "", member_file="5060.txt"),  # drama
    _gb(6790, "Demetrius--A-Play", "Demetrius A Play", "", "", member_file="6790.txt"),  # drama
    _gb(809, "Holiday-Romance", "Holiday Romance", "", ""),  # narrative
    _gb(7119, "The-Dolliver-Romance", "The Dolliver Romance", "", "", member_file="7119.txt"),  # narrative
    _gb(10707, "A-Christmas-MysteryThe-Story-of-Three-Wise-Men", "A Christmas MysteryThe Story of Three Wise Men", "", "", member_file="10707.txt"),  # narrative
    _gb(17581, "The-RomancersA-Comedy-in-Three-Acts", "The RomancersA Comedy in Three Acts", "", "", member_file="17581.txt"),  # narrative
    _gb(19367, "Romance--Two-Lectures", "Romance Two Lectures", "", "", member_file="19367.txt"),  # narrative
    _gb(21078, "The-Tale-of-Miss-Kitty-CatSlumber-Town-Tales", "The Tale of Miss Kitty CatSlumber Town Tales", "", "", member_file="21078.txt"),  # narrative
    _gb(21322, "The-Tale-of-Betsy-ButterflyTuck-Me-In-Tales", "The Tale of Betsy ButterflyTuck Me In Tales", "", "", member_file="21322.txt"),  # narrative
    _gb(21412, "The-Tale-of-Bobby-BobolinkTuck-me-In-Tales", "The Tale of Bobby BobolinkTuck me In Tales", "", "", member_file="21412.txt"),  # narrative
    _gb(21426, "The-Tale-of-Daddy-LonglegsTuck-Me-In-Tales", "The Tale of Daddy LonglegsTuck Me In Tales", "", "", member_file="21426.txt"),  # narrative
    _gb(21619, "The-Tale-of-Nimble-DeerSleepy-Time-Tales", "The Tale of Nimble DeerSleepy Time Tales", "", "", member_file="21619.txt"),  # narrative
    _gb(418, "A-Treatise-on-Good-Works", "A Treatise on Good Works", "", "", member_file="418.txt"),  # non-fiction
    _gb(908, "Treatise-on-Parents-and-Children", "Treatise on Parents and Children", "", "", member_file="908.txt"),  # non-fiction
    _gb(989, "Theologico-Political-Treatise-Part-1", "Theologico Political Treatise Part 1", "", "", member_file="989.txt"),  # non-fiction
    _gb(2791, "Essays-and-Tales", "Essays and Tales", "", "", member_file="2791.txt"),  # non-fiction
    _gb(2808, "Treatises-on-Friendship-and-Old-Age", "Treatises on Friendship and Old Age", "", "", member_file="2808.txt"),  # non-fiction
    _gb(3108, "Nine-Short-Essays", "Nine Short Essays", "", "", member_file="3108.txt"),  # non-fiction
    _gb(4754, "The-Philosophy-of-Despair", "The Philosophy of Despair", "", "", member_file="4754.txt"),  # non-fiction
    _gb(5329, "PoPHILO--Popular-Philosophy", "PoPHILO Popular Philosophy", "", "", member_file="5329.txt"),  # non-fiction
    _gb(6054, "Mrs-Caudle-s-Curtain-Lectures", "Mrs Caudle s Curtain Lectures", "", "", member_file="6054.txt"),  # non-fiction
    _gb(9850, "The-Lyric--An-Essay", "The Lyric An Essay", "", "", member_file="9850.txt"),  # non-fiction
    _gb(679, "Poems", "Poems", "", "", member_file="679.txt"),  # verse
    _gb(1050, "The-Dark-Lady-of-the-Sonnets", "The Dark Lady of the Sonnets", "", "", member_file="1050.txt"),  # verse
    _gb(1186, "Poems", "Poems", "", "", member_file="1186.txt"),  # verse
    _gb(1546, "Sonnets-on-Sundry-Notes-of-Music", "Sonnets on Sundry Notes of Music", "", "", member_file="1546.txt"),  # verse
    _gb(3305, "Ballads-of-Peace-in-War", "Ballads of Peace in War", "", "", member_file="3305.txt"),  # verse
    _gb(4756, "The-Love-Sonnets-of-a-Hoodlum", "The Love Sonnets of a Hoodlum", "", "", member_file="4756.txt"),  # verse
    _gb(8161, "Fragments-of-Ancient-Poetry", "Fragments of Ancient Poetry", "", "", member_file="8161.txt"),  # verse
    _gb(9093, "Rural-Tales-Ballads-and-Songs", "Rural Tales Ballads and Songs", "", "", member_file="9093.txt"),  # verse
    _gb(9542, "Graded-Poetry--Seventh-Year", "Graded Poetry Seventh Year", "", "", member_file="9542.txt"),  # verse
    _gb(9825, "A-Calendar-of-Sonnets", "A Calendar of Sonnets", "", "", member_file="9825.txt"),  # verse
)

GUTENBERG_HELDOUT = MultiDocCorpus(
    name="gutenberg-heldout-forms",
    description=(
        "Forty public-domain titles, ten per form, that no model in this "
        "project has trained on. An evaluation corpus for measuring transfer "
        "to unseen books by form; prepare it with --tokenizer-from pointed at "
        "the corpus whose models are being probed."
    ),
    # Computed 2026-10-21 over the assembled 40-document text, 4,047,093 chars.
    sha256="e5d295be9c62ef9c8118bd2fa9338f0e573258e5899d8a8bb9c6316d80f8ca9d",
    documents=GUTENBERG_HELDOUT_DOCS,
    text_license=GUTENBERG_MIXED.text_license,
    file_license=GUTENBERG_MIXED.file_license,
    restrictions=GUTENBERG_EXPANDED.restrictions,
    access_date="2026-10-21",
)

# 67 more non-narrative titles on top of the broad corpus, taking narrative
# prose from 67.8% of the bytes to 51.7%. Selected smallest-first per form
# from every verified-but-unused discovery pool; verse exhausts its pool
# (6.6 MB of a 9.6 MB budget), which is why the share lands at 51.7% rather
# than 50%. train_only, so evaluation still covers exactly the original 71.
#
# The question: broadening cost +1.19% at 1.77M/short clock and +0.28%
# (a tie) at 3.74M/long clock, at a 68%-narrative mixture. Does pushing to
# ~52% narrative stay free on seen books while buying more transfer to
# unseen ones (the held-out probe put transfer at -8.13%)? One more point
# on the mixture curve, at the size where the first point became free.
GUTENBERG_BROAD50_EXTRA_DOCS = (
    _gb(1008, "Divine-Comedy-Cary-s-Translation-Complete", "Divine Comedy Cary s Translation Complete", "", "", train_only=True, member_file="1008.txt"),  # drama
    _gb(1129, "The-Tragedy-of-Macbeth", "The Tragedy of Macbeth", "", "", train_only=True, member_file="1129.txt"),  # drama
    _gb(1131, "The-Tragedy-of-Coriolanus", "The Tragedy of Coriolanus", "", "", train_only=True, member_file="1131.txt"),  # drama
    _gb(1192, "The-Old-Bachelor--a-Comedy", "The Old Bachelor a Comedy", "", "", train_only=True),  # drama
    _gb(1244, "Love-for-Love--a-Comedy", "Love for Love a Comedy", "", "", train_only=True),  # drama
    _gb(1684, "The-Egoist--A-Comedy-in-Narrative", "The Egoist A Comedy in Narrative", "", "", train_only=True, member_file="1684.txt"),  # drama
    _gb(1995, "The-Divine-Comedy-Volume-1-Hell", "The Divine Comedy Volume 1 Hell", "", "", train_only=True, member_file="1995.txt"),  # drama
    _gb(1996, "The-Divine-Comedy-Volume-2-Purgatory", "The Divine Comedy Volume 2 Purgatory", "", "", train_only=True, member_file="1996.txt"),  # drama
    _gb(2920, "Four-Short-Plays", "Four Short Plays", "", "", train_only=True, member_file="2920.txt"),  # drama
    _gb(3702, "Foul-Play", "Foul Play", "", "", train_only=True, member_file="3702.txt"),  # drama
    _gb(5055, "Plays---First-Series", "Plays First Series", "", "", train_only=True, member_file="5055.txt"),  # drama
    _gb(5056, "Plays---Second-Series", "Plays Second Series", "", "", train_only=True, member_file="5056.txt"),  # drama
    _gb(5057, "Plays---Third-Series", "Plays Third Series", "", "", train_only=True, member_file="5057.txt"),  # drama
    _gb(5058, "Plays---Fourth-Series", "Plays Fourth Series", "", "", train_only=True, member_file="5058.txt"),  # drama
    _gb(5059, "Plays---Fifth-Series", "Plays Fifth Series", "", "", train_only=True, member_file="5059.txt"),  # drama
    _gb(5435, "The-Stillwater-Tragedy", "The Stillwater Tragedy", "", "", train_only=True, member_file="5435.txt"),  # drama
    _gb(6784, "Love-and-Intrigue--A-Tragedy", "Love and Intrigue A Tragedy", "", "", train_only=True, member_file="6784.txt"),  # drama
    _gb(6786, "The-Piccolomini--A-Play", "The Piccolomini A Play", "", "", train_only=True, member_file="6786.txt"),  # drama
    _gb(6791, "Mary-Stuart--A-Tragedy", "Mary Stuart A Tragedy", "", "", train_only=True, member_file="6791.txt"),  # drama
    _gb(7487, "Count-Alarcos--a-Tragedy", "Count Alarcos a Tragedy", "", "", train_only=True),  # drama
    _gb(7840, "The-Female-Gamester--A-Tragedy", "The Female Gamester A Tragedy", "", "", train_only=True, member_file="7840.txt"),  # drama
    _gb(8464, "Rest-Harrow--A-Comedy-of-Resolution", "Rest Harrow A Comedy of Resolution", "", "", train_only=True),  # drama
    _gb(8771, "Jurgen--A-Comedy-of-Justice", "Jurgen A Comedy of Justice", "", "", train_only=True, member_file="8771.txt"),  # drama
    _gb(9161, "Comedy-of-Marriage-and-Other-Tales", "Comedy of Marriage and Other Tales", "", "", train_only=True),  # drama
    _gb(9663, "Domnei--A-Comedy-of-Woman-Worship", "Domnei A Comedy of Woman Worship", "", "", train_only=True, member_file="9663.txt"),  # drama
    _gb(627, "Essays-of-Travel", "Essays of Travel", "", "", train_only=True),  # non-fiction
    _gb(1360, "Historical-Lectures-and-Essays", "Historical Lectures and Essays", "", "", train_only=True, member_file="1360.txt"),  # non-fiction
    _gb(2010, "The-Autobiography-of-Charles-Darwin", "The Autobiography of Charles Darwin", "", "", train_only=True, member_file="2010.txt"),  # non-fiction
    _gb(2162, "Anarchism-and-Other-Essays", "Anarchism and Other Essays", "", "", train_only=True, member_file="2162.txt"),  # non-fiction
    _gb(2169, "Miscellaneous-Writings-and-Speeches---Volume-3", "Miscellaneous Writings and Speeches Volume 3", "", "", train_only=True, member_file="2169.txt"),  # non-fiction
    _gb(2523, "The-Memoirs-of-Victor-Hugo", "The Memoirs of Victor Hugo", "", "", train_only=True),  # non-fiction
    _gb(2944, "Essays---First-Series", "Essays First Series", "", "", train_only=True, member_file="2944.txt"),  # non-fiction
    _gb(3175, "Mark-Twain-s-Burlesque-Autobiography", "Mark Twain s Burlesque Autobiography", "", "", train_only=True),  # non-fiction
    _gb(3188, "Mark-Twain-s-Speeches", "Mark Twain s Speeches", "", "", train_only=True),  # non-fiction
    _gb(3559, "Memoirs-of-Napoleon-Bonaparte-Volume-09", "Memoirs of Napoleon Bonaparte Volume 09", "", "", train_only=True, member_file="3559.txt"),  # non-fiction
    _gb(4087, "An-Essay-Upon-Projects", "An Essay Upon Projects", "", "", train_only=True),  # non-fiction
    _gb(4391, "Selections-from-the-Principles-of-Philosophy", "Selections from the Principles of Philosophy", "", "", train_only=True, member_file="4391.txt"),  # non-fiction
    _gb(5717, "Bergson-and-His-Philosophy", "Bergson and His Philosophy", "", "", train_only=True, member_file="5717.txt"),  # non-fiction
    _gb(5827, "The-Problems-of-Philosophy", "The Problems of Philosophy", "", "", train_only=True, member_file="5827.txt"),  # non-fiction
    _gb(6015, "Captain-Macklin--His-Memoirs", "Captain Macklin His Memoirs", "", "", train_only=True),  # non-fiction
    _gb(6031, "Memoirs-of-My-Life-and-Writings", "Memoirs of My Life and Writings", "", "", train_only=True, member_file="6031.txt"),  # non-fiction
    _gb(6314, "Biographical-Essays", "Biographical Essays", "", "", train_only=True, member_file="6314.txt"),  # non-fiction
    _gb(6435, "The-Principles-of-Scientific-Management", "The Principles of Scientific Management", "", "", train_only=True, member_file="6435.txt"),  # non-fiction
    _gb(7976, "Mr.-Dooley-s-Philosophy", "Mr. Dooley s Philosophy", "", "", train_only=True),  # non-fiction
    _gb(8108, "Literary-and-Social-Essays", "Literary and Social Essays", "", "", train_only=True, member_file="8108.txt"),  # non-fiction
    _gb(9060, "The-Public-Orations-of-Demosthenes-volume-1", "The Public Orations of Demosthenes volume 1", "", "", train_only=True, member_file="9060.txt"),  # non-fiction
    _gb(10378, "Autobiography", "Autobiography", "", "", train_only=True, member_file="10378.txt"),  # non-fiction
    _gb(10708, "Miscellaneous-Essays", "Miscellaneous Essays", "", "", train_only=True, member_file="10708.txt"),  # non-fiction
    _gb(11391, "Lectures-on-Art", "Lectures on Art", "", "", train_only=True, member_file="11391.txt"),  # non-fiction
    _gb(11945, "Essays-of-Schopenhauer", "Essays of Schopenhauer", "", "", train_only=True),  # non-fiction
    _gb(11984, "A-Pluralistic-Universe--13-Hibbert-Lectures-at-Manchester-College-on-the-Present-Situation-in_", "A Pluralistic Universe 13 Hibbert Lectures at Manchester College on the Present Situation in", "", "", train_only=True, member_file="11984.txt"),  # non-fiction
    _gb(259, "Ballads-of-a-Cheechako", "Ballads of a Cheechako", "", "", train_only=True, member_file="259.txt"),  # verse
    _gb(617, "Poems", "Poems", "", "", train_only=True, member_file="617.txt"),  # verse
    _gb(1054, "A-Collection-of-Ballads", "A Collection of Ballads", "", "", train_only=True),  # verse
    _gb(1365, "The-Complete-Poetical-Works-of-Henry-Wadsworth-Longfellow", "The Complete Poetical Works of Henry Wadsworth Longfellow", "", "", train_only=True, member_file="1365.txt"),  # verse
    _gb(2831, "A-Bundle-of-Ballads", "A Bundle of Ballads", "", "", train_only=True, member_file="2831.txt"),  # verse
    _gb(5332, "The-Love-Sonnets-of-a-Car-Conductor", "The Love Sonnets of a Car Conductor", "", "", train_only=True, member_file="5332.txt"),  # verse
    _gb(6844, "The-Poetical-Works-of-Mrs.-Leprohon", "The Poetical Works of Mrs. Leprohon", "", "", train_only=True, member_file="6844.txt"),  # verse
    _gb(7400, "The-Poetical-Works-of-Oliver-Wendell-Holmes---Complete", "The Poetical Works of Oliver Wendell Holmes Complete", "", "", train_only=True, member_file="7400.txt"),  # verse
    _gb(7796, "Poems", "Poems", "", "", train_only=True, member_file="7796.txt"),  # verse
    _gb(8775, "Poems", "Poems", "", "", train_only=True, member_file="8775.txt"),  # verse
    _gb(9484, "Georgian-Poetry-1911-1912", "Georgian Poetry 1911 1912", "", "", train_only=True, member_file="9484.txt"),  # verse
    _gb(9506, "Georgian-Poetry-1913-15", "Georgian Poetry 1913 15", "", "", train_only=True, member_file="9506.txt"),  # verse
    _gb(9546, "Georgian-Poetry-1916-1917", "Georgian Poetry 1916 1917", "", "", train_only=True, member_file="9546.txt"),  # verse
    _gb(9621, "Georgian-Poetry-1918-19", "Georgian Poetry 1918 19", "", "", train_only=True, member_file="9621.txt"),  # verse
    _gb(9640, "Georgian-Poetry-1920-22", "Georgian Poetry 1920 22", "", "", train_only=True, member_file="9640.txt"),  # verse
    _gb(10219, "The-Poetical-Works-of-William-Wordsworth---Volume-1", "The Poetical Works of William Wordsworth Volume 1", "", "", train_only=True, member_file="10219.txt"),  # verse
)

GUTENBERG_BROAD50 = MultiDocCorpus(
    name="gutenberg-mixed-broad50",
    description=(
        "The broad corpus broadened again: 218 titles, narrative prose at "
        "51.7% of the bytes. The additions are train_only, so it evaluates "
        "on exactly the broad arms' splits. Prepare with --tokenizer-from "
        "pointed at the broad arm."
    ),
    sha256="",
    documents=GUTENBERG_BROAD_DOCS_FULL + GUTENBERG_BROAD50_EXTRA_DOCS,
    text_license=GUTENBERG_MIXED.text_license,
    file_license=GUTENBERG_MIXED.file_license,
    restrictions=GUTENBERG_EXPANDED.restrictions,
    access_date="2026-10-21",
)


CORPORA = {
    TINY_SHAKESPEARE.name: TINY_SHAKESPEARE,
    GUTENBERG_MIXED.name: GUTENBERG_MIXED,
    GUTENBERG_EXPANDED.name: GUTENBERG_EXPANDED,
    GUTENBERG_71_FULL.name: GUTENBERG_71_FULL,
    GUTENBERG_71_CTRL.name: GUTENBERG_71_CTRL,
    GUTENBERG_BROAD_FULL.name: GUTENBERG_BROAD_FULL,
    GUTENBERG_BROAD_CTRL.name: GUTENBERG_BROAD_CTRL,
    GUTENBERG_NARRATIVE.name: GUTENBERG_NARRATIVE,
    GUTENBERG_HELDOUT.name: GUTENBERG_HELDOUT,
    GUTENBERG_BROAD50.name: GUTENBERG_BROAD50,
}


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch_corpus(corpus: Corpus, cache_dir: str | Path, local_file: str | Path | None = None) -> Path:
    """Download (or copy) the corpus and verify its checksum before use."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{corpus.name}.txt"

    if target.exists() and sha256_bytes(target.read_bytes()) == corpus.sha256:
        return target

    payload: bytes | None = None
    if local_file is not None:
        payload = Path(local_file).read_bytes()
    else:
        errors = []
        for url in corpus.urls:
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    blob = resp.read()
                if url.endswith((".tar.gz", "tar.gz/refs/heads/master")) or "codeload" in url:
                    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                        member = tar.extractfile(corpus.member)  # type: ignore[arg-type]
                        if member is None:
                            raise RuntimeError(f"{corpus.member} missing from archive at {url}")
                        blob = member.read()
                payload = blob
                break
            except Exception as exc:  # pragma: no cover - network dependent
                errors.append(f"{url}: {exc}")
        if payload is None:
            raise RuntimeError("could not fetch corpus:\n  " + "\n  ".join(errors))

    digest = sha256_bytes(payload)
    if digest != corpus.sha256:
        raise RuntimeError(
            f"checksum mismatch for {corpus.name}: expected {corpus.sha256}, got {digest}. "
            "Refusing to train on an unverified corpus."
        )
    target.write_bytes(payload)
    return target


# --------------------------------------------------------------------------
# Multi-document acquisition
# --------------------------------------------------------------------------

# Project Gutenberg wraps every text in a header and a footer. Both have been
# worded several ways over the years; these cover the forms in this corpus.
PG_START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S
)
PG_END = re.compile(
    r"(\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"
    r"|End of (?:the )?Project Gutenberg(?:'s)? EBook)", re.I | re.S
)
PG_PRODUCED_BY = re.compile(r"^\s*(Produced by|E-?text prepared by|Transcribed from).*?(?:\n\s*\n)", re.I | re.S)


def strip_gutenberg_boilerplate(text: str) -> tuple[str, dict[str, int]]:
    """Remove PG's header, footer and transcriber credits.

    Three reasons, in order of importance: (1) the boilerplate is the one part
    of the file that is not public domain, and PG asks that it be removed if
    the text is used as an ordinary public-domain work; (2) it is identical
    across every book, so leaving it in creates thousands of duplicate lines
    that teach the model to emit licence text; (3) it is not the register we
    are trying to model.
    """
    original = len(text)
    m = PG_START.search(text)
    head = m.end() if m else 0
    m2 = PG_END.search(text, head)
    tail = m2.start() if m2 else len(text)
    body = text[head:tail]
    body = PG_PRODUCED_BY.sub("", body, count=1)
    body = body.strip("\n") + "\n"
    return body, {"original_chars": original, "body_chars": len(body), "removed_chars": original - len(body)}


def fetch_documents(
    corpus: MultiDocCorpus, cache_dir: str | Path
) -> tuple[str, list[dict]]:
    """Download each document, strip boilerplate, and verify the corpus hash.

    Documents are cached individually so a re-run costs nothing, and the
    assembled corpus is checksummed as a whole: if any document changes
    upstream, training refuses to start.
    """
    cache_dir = Path(cache_dir) / corpus.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    bodies: list[str] = []

    for doc in corpus.documents:
        cached = cache_dir / f"{doc.doc_id}.txt"
        # The boilerplate statistics are provenance, not debug output: they are
        # the evidence that the licence header was removed from this document
        # and how much of it there was. Recomputing them from the cached body is
        # impossible -- the header is gone -- so they are cached beside it.
        # Without this the manifest silently degraded to {"cached": true} on
        # every run after the first, which is every run that matters.
        stats_path = cache_dir / f"{doc.doc_id}.stats.json"
        if cached.exists() and stats_path.exists():
            body = cached.read_text(encoding="utf-8")
            strip_stats = json.loads(stats_path.read_text())
            strip_stats["from_cache"] = True
        else:
            url = f"https://codeload.github.com/{doc.repo}/tar.gz/refs/heads/master"
            with urllib.request.urlopen(url, timeout=120) as resp:
                blob = resp.read()
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                member = None
                for candidate in (doc.member, doc.member.replace("-0.txt", ".txt")):
                    try:
                        member = tar.extractfile(candidate)
                    except KeyError:
                        member = None
                    if member is not None:
                        break
                if member is None:  # fall back to the largest .txt in the archive
                    txts = [t for t in tar.getmembers() if t.name.endswith(".txt")]
                    if not txts:
                        raise RuntimeError(f"no .txt in {doc.repo}")
                    member = tar.extractfile(max(txts, key=lambda t: t.size))
                raw = member.read()  # type: ignore[union-attr]
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            body, strip_stats = strip_gutenberg_boilerplate(text)
            cached.write_text(body, encoding="utf-8")
            stats_path.write_text(json.dumps(strip_stats, indent=2) + "\n")
        bodies.append(body)
        records.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "author": doc.author,
                "year": doc.year,
                "source_url": f"https://codeload.github.com/{doc.repo}/tar.gz/refs/heads/master",
                "member": doc.member,
                "chars": len(body),
                "bytes": len(body.encode("utf-8")),
                "sha256": sha256_bytes(body.encode("utf-8")),
                "boilerplate": strip_stats,
            }
        )

    assembled = "\n\n".join(bodies)
    digest = sha256_bytes(assembled.encode("utf-8"))
    if corpus.sha256 and digest != corpus.sha256:
        raise RuntimeError(
            f"checksum mismatch for {corpus.name}: expected {corpus.sha256}, got {digest}. "
            "Refusing to train on an unverified corpus. If this is a deliberate change to "
            "the document list, update the registry hash in src/data.py."
        )
    if not corpus.sha256:
        print(f"NOTE: {corpus.name} has no pinned hash. Computed sha256={digest}")
    return assembled, records


# --------------------------------------------------------------------------
# Auditing
# --------------------------------------------------------------------------

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone_like": re.compile(r"(?<!\d)(?:\+\d{1,3}[ -]?)?(?:\(\d{2,4}\)[ -]?)?\d{3}[ -]\d{4}(?!\d)"),
    "ipv4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "url": re.compile(r"https?://\S+"),
    "aws_key_like": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key_header": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def scan_sensitive(text: str) -> dict[str, int]:
    """Count obvious personal data / secret patterns. Cheap, and it is a gate.

    A clean result is not proof of safety; it is proof that the easy cases were
    checked, which is the claim the model card is allowed to make.
    """
    return {name: len(pattern.findall(text)) for name, pattern in PII_PATTERNS.items()}


def duplicate_stats(text: str) -> dict[str, float | int]:
    """Exact-duplicate rate over blank-line separated blocks."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    unique = set(blocks)
    dup_chars = sum(len(b) for b in blocks) - sum(len(b) for b in unique)
    return {
        "blocks": len(blocks),
        "unique_blocks": len(unique),
        "duplicate_blocks": len(blocks) - len(unique),
        "duplicate_block_fraction": (len(blocks) - len(unique)) / max(len(blocks), 1),
        "duplicate_chars": dup_chars,
    }


def git_revision(repo_root: str | Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------


@dataclass
class PreparedData:
    out_dir: Path
    manifest: dict
    tokenizer_path: Path
    split_paths: dict[str, Path] = field(default_factory=dict)


def prepare(
    corpus_name: str = "tiny-shakespeare",
    tokenizer_kind: str = "char",
    vocab_size: int = 512,
    out_dir: str | Path = "data/prepared",
    cache_dir: str | Path = "data/raw",
    splits: tuple[float, float, float] = (0.90, 0.05, 0.05),
    local_file: str | Path | None = None,
    repo_root: str | Path = ".",
    tokenizer_sample_chars: int = 4_000_000,
    tokenizer_from: str | Path | None = None,
) -> PreparedData:
    """Fetch, split, tokenize and write a corpus.

    ``tokenizer_from`` points at an already-prepared directory (or a
    ``tokenizer.json`` directly) and reuses that vocabulary instead of training
    a new one. Bits per byte is tokenizer-independent as a *unit*, but a model
    is not: its embedding table is indexed by token id, so two models can only
    be scored on the same text if that text was tokenized the same way. Without
    this, any comparison across two corpora is confounded by the vocabulary,
    and whether the two vocabularies happen to coincide depends on invisible
    details like the order of the documents and the sample cap.
    """
    if abs(sum(splits) - 1.0) > 1e-9:
        raise ValueError(f"splits must sum to 1.0, got {splits}")
    corpus = CORPORA[corpus_name]
    multi_doc = isinstance(corpus, MultiDocCorpus)

    doc_records: list[dict] = []
    if multi_doc:
        text, doc_records = fetch_documents(corpus, cache_dir)
        documents = [d["chars"] for d in doc_records]
    else:
        raw_path = fetch_corpus(corpus, cache_dir, local_file=local_file)
        text = raw_path.read_text(encoding="utf-8")
        documents = [len(text)]

    n = len(text)

    def snap(s: str, i: int) -> int:
        """Move a split boundary to the next blank line, never mid-paragraph."""
        j = s.find("\n\n", i)
        return (j + 2) if j != -1 else len(s)

    # Each document is split contiguously and in the same proportions, then the
    # pieces are concatenated per split. Splitting *whole documents* into
    # different splits would be cleaner for leakage but would make validation a
    # different set of authors from training, measuring domain transfer instead
    # of language modelling. Splitting within a document and keeping the pieces
    # contiguous avoids both the leak and the author shift.
    parts: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    boundaries: list[list[int]] = []
    offset = 0
    train_only_flags = ([getattr(d, "train_only", False) for d in corpus.documents]
                        if multi_doc else [False])
    eval_only_flags = ([getattr(d, "eval_only", False) for d in corpus.documents]
                       if multi_doc else [False])
    # Parallel to parts["train"], not to `documents`. With eval-only documents
    # the two lengths differ, and zipping the per-document flags against the
    # per-piece list would silently misalign the tokenizer sample below.
    train_piece_train_only: list[bool] = []
    for length, train_only, eval_only in zip(documents, train_only_flags, eval_only_flags):
        doc = text[offset : offset + length]
        offset += length + (2 if multi_doc else 0)  # the "\n\n" join
        if train_only:
            # No val or test piece at all -- not an empty one. An empty piece
            # would still insert an EOS separator into those splits and change
            # their bytes, which is exactly what this flag exists to prevent.
            parts["train"].append(doc)
            train_piece_train_only.append(True)
            boundaries.append([len(doc), len(doc)])
            continue
        i_train = snap(doc, int(len(doc) * splits[0]))
        i_val = snap(doc, int(len(doc) * (splits[0] + splits[1])))
        # `snap` only ever moves forward, so on a document with few paragraph
        # breaks both boundaries can land on the same final break and silently
        # empty the val or test piece -- a document that contributes to
        # training but to no evaluation. Caught by tests/test_multidoc.py.
        # Long documents never hit this, so the 27 books here are unaffected;
        # the fallback keeps a short one honest rather than dropping it.
        if not 0 < i_train < i_val < len(doc):
            i_train = int(len(doc) * splits[0])
            i_val = int(len(doc) * (splits[0] + splits[1]))
        if not eval_only:
            # The boundaries are a function of len(doc) and `splits` alone, so
            # dropping the training piece leaves the val and test pieces exactly
            # where a full-corpus preparation would put them. That is what makes
            # the control corpus share its evaluation splits byte for byte.
            parts["train"].append(doc[:i_train])
            train_piece_train_only.append(False)
        parts["val"].append(doc[i_train:i_val])
        parts["test"].append(doc[i_val:])
        boundaries.append([i_train, i_val])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer is trained on the training split only, and on at most
    # `tokenizer_sample_chars` of it: BPE merge counting is O(corpus x merges),
    # and the vocabulary a few MB produces is indistinguishable from the one the
    # whole corpus produces. The cap is recorded, because an unrecorded sample
    # is an unreproducible tokenizer.
    train_text = "".join(parts["train"])
    # The tokenizer is trained on the *evaluation-bearing* documents only --
    # those that contribute to val and test. Train-only documents are excluded
    # from the sample even though they are trained on.
    #
    # Without this, adding train-only text silently rebuilds the vocabulary,
    # and a run on the larger corpus then differs from the smaller one in two
    # ways instead of one. The flag exists to make exactly one thing change,
    # so the vocabulary has to be a function of the fixed part of the corpus.
    # Caught by tests/test_train_only.py, which failed on token hashes while
    # the underlying text was byte-identical.
    #
    # The tradeoff is real and deliberate: a 3x larger corpus might deserve a
    # larger or better-fitted vocabulary, and this gives up that gain. That is
    # a separate experiment -- vocabulary size -- and conflating the two would
    # make neither answerable.
    #
    # This applies to byte-level tokenizers only. A character tokenizer's
    # vocabulary *is* the set of characters in its sample, so excluding text
    # it must later encode makes encoding fail outright -- which is how this
    # restriction was found. A byte-BPE vocabulary always covers all 256
    # bytes, so unseen text costs compression, never correctness.
    excludes_train_only = tokenizer_kind != "char" and any(train_only_flags)
    sample_source = train_text
    if excludes_train_only:
        sample_source = "".join(piece for piece, flag in zip(parts["train"], train_piece_train_only)
                                if not flag) or train_text
    sample = sample_source[:tokenizer_sample_chars] if tokenizer_sample_chars else sample_source

    reused_from: str | None = None
    if tokenizer_from is not None:
        src = Path(tokenizer_from)
        src_file = src / "tokenizer.json" if src.is_dir() else src
        if not src_file.is_file():
            raise FileNotFoundError(
                f"tokenizer_from={tokenizer_from!r} does not name a tokenizer.json "
                f"(looked for {src_file})"
            )
        tokenizer = BaseTokenizer.load(src_file)
        if tokenizer.kind != tokenizer_kind:
            raise ValueError(
                f"tokenizer_from holds a {tokenizer.kind!r} tokenizer but "
                f"tokenizer_kind={tokenizer_kind!r} was requested; reusing a vocabulary "
                "only makes sense when it is the same kind of vocabulary"
            )
        if tokenizer_kind != "char" and tokenizer.vocab_size != vocab_size:
            raise ValueError(
                f"tokenizer_from holds a vocabulary of {tokenizer.vocab_size} but "
                f"vocab_size={vocab_size} was requested"
            )
        reused_from = str(src_file)
        sample = ""
    else:
        tokenizer = build_tokenizer(tokenizer_kind, sample, vocab_size=vocab_size)
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32
    split_paths: dict[str, Path] = {}
    split_stats: dict[str, dict] = {}
    for name, pieces in parts.items():
        # array.array rather than a list: a Python int object costs ~28 bytes
        # plus 8 for the list slot, so a 20M-token split needs ~700 MB as a
        # list and 80 MB here. That is the difference between preparing a
        # 60 MB corpus on a 3 GB machine and being killed by the OOM reaper.
        token_ids = array.array("i")
        for i, piece in enumerate(pieces):
            token_ids.extend(tokenizer.encode(piece))
            if multi_doc and i < len(pieces) - 1:
                # Mark the document boundary so the model can learn that text
                # ends. EOS encodes zero bytes, so it is excluded from both
                # sides of the bits-per-byte ratio.
                token_ids.append(tokenizer.eos_id)
        ids = np.array(token_ids, dtype=dtype)
        path = out_dir / f"{name}.bin"
        ids.tofile(path)
        joined = "".join(pieces)
        split_paths[name] = path
        split_stats[name] = {
            "chars": len(joined),
            "bytes": len(joined.encode("utf-8")),
            "tokens": int(ids.size),
            "documents": len(pieces),
            "eos_separators": (len(pieces) - 1) if multi_doc else 0,
            "tokens_per_char": float(ids.size / max(len(joined), 1)),
            "sha256_text": sha256_bytes(joined.encode("utf-8")),
            "sha256_tokens": sha256_bytes(ids.tobytes()),
            "file": str(path.name),
            "dtype": np.dtype(dtype).name,
        }

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_revision": git_revision(repo_root),
        "corpus": {
            "name": corpus.name,
            "description": corpus.description,
            "source_urls": (
                [d["source_url"] for d in doc_records] if multi_doc else list(corpus.urls)
            ),
            "member": None if multi_doc else corpus.member,
            "access_date": corpus.access_date,
            "sha256": corpus.sha256,
            "text_license": corpus.text_license,
            "file_license": corpus.file_license,
            "restrictions": corpus.restrictions,
            "chars": n,
            "bytes": len(text.encode("utf-8")),
            "documents": len(documents),
            "document_records": doc_records,
        },
        "split_policy": {
            "method": (
                "per document, contiguous, boundaries snapped to a blank line"
                if multi_doc
                else "contiguous, document order, boundaries snapped to a blank line"
            ),
            "fractions": list(splits),
            "rationale": (
                "A shuffled split leaks overlapping context windows across the boundary. "
                "Splitting each document contiguously and in the same proportions avoids "
                "that leak and also keeps every author present in every split, so "
                "validation measures language modelling rather than domain transfer."
            ),
            "boundaries_chars": boundaries[0] if len(boundaries) == 1 else boundaries,
        },
        "splits": split_stats,
        "tokenizer": {
            "kind": tokenizer_kind,
            "vocab_size": tokenizer.vocab_size,
            "trained_on": "train split only" if reused_from is None else f"reused from {reused_from}",
            "reused_from": reused_from,
            "requested_vocab_size": vocab_size if tokenizer_kind != "char" else None,
            "bos_id": tokenizer.bos_id,
            "eos_id": tokenizer.eos_id,
            "sha256": sha256_bytes(tokenizer_path.read_bytes()),
            "trained_on_chars": len(sample),
            "train_split_chars": len(train_text),
            "measurement_on_val": tokenizer.measure("".join(parts["val"])),
        },
        "audit": {
            "sensitive_patterns_full_corpus": scan_sensitive(text),
            "duplicates_full_corpus": duplicate_stats(text),
            "note": (
                "Project Gutenberg header/footer boilerplate is removed before this scan; "
                "what remains is body text. No further records were removed. The duplicate "
                "rate is reported so the decision is visible."
                if multi_doc
                else "No records were removed. The duplicate rate is reported so the decision "
                "is visible; repeated short lines in dialogue are a property of the text, "
                "not scrape noise."
            ),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PreparedData(out_dir=out_dir, manifest=manifest, tokenizer_path=tokenizer_path, split_paths=split_paths)


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


class TokenBatcher:
    """Deterministic batches of ``[batch_size, block_size]`` token windows.

    Batch contents are a pure function of ``(seed, step)``. That makes a resumed
    run see exactly the data an uninterrupted run would have seen at that step,
    which is the property ``tests/test_resume.py`` checks, and it means the data
    position in a checkpoint is just the step number.
    """

    def __init__(self, path: str | Path, block_size: int, batch_size: int, seed: int = 1337) -> None:
        self.path = Path(path)
        self.block_size = block_size
        self.batch_size = batch_size
        self.seed = seed
        self.dtype = np.uint16
        self.data = np.memmap(self.path, dtype=self.dtype, mode="r")
        if len(self.data) < block_size + 1:
            raise ValueError(
                f"{self.path} holds {len(self.data)} tokens, need at least block_size+1={block_size + 1}"
            )

    def __len__(self) -> int:
        return len(self.data)

    @property
    def tokens_per_batch(self) -> int:
        return self.batch_size * self.block_size

    def batch(self, step: int, device: str = "cpu"):
        import torch

        rng = np.random.default_rng([self.seed, step])
        hi = len(self.data) - self.block_size - 1
        starts = rng.integers(0, hi, size=self.batch_size)
        x = np.stack([self.data[s : s + self.block_size] for s in starts]).astype(np.int64)
        y = np.stack([self.data[s + 1 : s + 1 + self.block_size] for s in starts]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    def iter_eval_batches(self, max_tokens: int | None = None, device: str = "cpu",
                          spread: bool = False) -> Iterator:
        """Non-overlapping windows covering the split, in order.

        Deterministic and exhaustive, which is what an evaluation number needs
        to be comparable between runs.

        ``max_tokens`` caps the cost by taking a *prefix* of the split. On a
        single-document corpus that is a fair sample. On a multi-document one
        it is not: the splits are the per-document pieces concatenated in
        document order, so a prefix covers the first few documents and nothing
        else. With the 27-book corpus and the default 98,304-token budget the
        in-training validation metric saw roughly the first eight books -- a
        third of the split, and not a random third.

        ``spread=True`` keeps the same cost and covers the whole split by
        taking evenly spaced windows instead of the leading ones. It is off by
        default so that every previously published number stays reproducible;
        new configurations should turn it on, and must not be compared against
        numbers produced without it.
        """
        import torch

        T = self.block_size
        n = len(self.data) - 1
        total_windows = n // T
        if max_tokens is None:
            window_ids = list(range(total_windows))
        else:
            want = max(1, min(total_windows, max_tokens // T))
            window_ids = ([(j * total_windows) // want for j in range(want)] if spread
                          else list(range(want)))
        for i in range(0, len(window_ids), self.batch_size):
            starts = [j * T for j in window_ids[i : i + self.batch_size]]
            x = np.stack([self.data[s : s + T] for s in starts]).astype(np.int64)
            y = np.stack([self.data[s + 1 : s + 1 + T] for s in starts]).astype(np.int64)
            yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def load_tokenizer(prepared_dir: str | Path) -> BaseTokenizer:
    return BaseTokenizer.load(Path(prepared_dir) / "tokenizer.json")


def load_manifest(prepared_dir: str | Path) -> dict:
    return json.loads((Path(prepared_dir) / "manifest.json").read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Prepare a corpus: fetch, audit, split, tokenize.")
    ap.add_argument("--corpus", default="tiny-shakespeare", choices=sorted(CORPORA))
    ap.add_argument("--tokenizer", default="char", choices=["char", "byte_bpe"])
    ap.add_argument("--vocab-size", type=int, default=512, help="byte_bpe only (includes 256 bytes + 2 specials)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cache-dir", default="data/raw")
    ap.add_argument("--local-file", default=None, help="use a local copy instead of downloading")
    ap.add_argument("--tokenizer-sample-chars", type=int, default=4_000_000,
                    help="cap the text a BPE vocabulary is fitted on; 0 means the whole train split")
    ap.add_argument("--tokenizer-from", default=None,
                    help="reuse the vocabulary of an already-prepared directory instead of "
                         "fitting a new one; required to compare models across two corpora")
    args = ap.parse_args()

    out_dir = args.out_dir or f"data/prepared-{args.tokenizer}{'' if args.tokenizer == 'char' else '-' + str(args.vocab_size)}"
    prepared = prepare(
        corpus_name=args.corpus,
        tokenizer_kind=args.tokenizer,
        vocab_size=args.vocab_size,
        out_dir=out_dir,
        cache_dir=args.cache_dir,
        local_file=args.local_file,
        repo_root=Path(__file__).resolve().parents[3],
        tokenizer_sample_chars=args.tokenizer_sample_chars,
        tokenizer_from=args.tokenizer_from,
    )
    m = prepared.manifest
    print(f"corpus      {m['corpus']['name']}  {m['corpus']['chars']:,} chars in "
          f"{m['corpus'].get('documents', 1)} document(s)  sha256 {m['corpus']['sha256'][:12] or '(unpinned)'}")
    print(f"tokenizer   {m['tokenizer']['kind']}  vocab {m['tokenizer']['vocab_size']}  "
          f"{m['tokenizer']['measurement_on_val']['tokens_per_char']:.3f} tokens/char on val")
    for name, s in m["splits"].items():
        print(f"  {name:<5} {s['tokens']:>9,} tokens  {s['bytes']:>9,} bytes -> {s['file']}")
    print(f"audit       sensitive={m['audit']['sensitive_patterns_full_corpus']}")
    print(f"            duplicate blocks={m['audit']['duplicates_full_corpus']['duplicate_blocks']}"
          f" / {m['audit']['duplicates_full_corpus']['blocks']}")
    print(f"written to  {prepared.out_dir}")


if __name__ == "__main__":
    main()
