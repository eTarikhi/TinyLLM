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

import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass, field
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


def _gb(pg: int, slug: str, title: str, author: str, year: str) -> Document:
    """GITenberg mirrors Project Gutenberg as one git repository per book.

    Each repo carries ``<id>-0.txt`` (UTF-8) and ``<id>.txt`` (legacy encoding);
    we take the UTF-8 one. The repos were last pushed in 2018, so a tarball of
    ``master`` is a stable snapshot -- which is why this is reproducible and a
    live gutenberg.org fetch would not be.
    """
    return Document(
        doc_id=f"pg{pg}",
        title=title,
        author=author,
        year=year,
        repo=f"GITenberg/{slug}_{pg}",
        member=f"{slug}_{pg}-master/{pg}-0.txt",
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

CORPORA = {TINY_SHAKESPEARE.name: TINY_SHAKESPEARE, GUTENBERG_MIXED.name: GUTENBERG_MIXED}


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
) -> PreparedData:
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
    for length in documents:
        doc = text[offset : offset + length]
        offset += length + (2 if multi_doc else 0)  # the "\n\n" join
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
        parts["train"].append(doc[:i_train])
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
    sample = train_text[:tokenizer_sample_chars] if tokenizer_sample_chars else train_text
    tokenizer = build_tokenizer(tokenizer_kind, sample, vocab_size=vocab_size)
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32
    split_paths: dict[str, Path] = {}
    split_stats: dict[str, dict] = {}
    for name, pieces in parts.items():
        token_ids: list[int] = []
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
            "trained_on": "train split only",
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

    def iter_eval_batches(self, max_tokens: int | None = None, device: str = "cpu") -> Iterator:
        """Non-overlapping windows covering the split, in order.

        Deterministic and exhaustive, which is what an evaluation number needs
        to be comparable between runs.
        """
        import torch

        T = self.block_size
        n = len(self.data) - 1
        if max_tokens is not None:
            n = min(n, max_tokens)
        n_windows = n // T
        for i in range(0, n_windows, self.batch_size):
            starts = [j * T for j in range(i, min(i + self.batch_size, n_windows))]
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
