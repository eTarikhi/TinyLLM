# What the corpus is made of, and what to build next

Every hyperparameter this project has tested is now flat or harmful. The
largest number anywhere in the results is not a lever at all: across the 71
test books the released model spans **1.3511 to 2.1509 bits per byte**, a
range of 50.6% of the mean, against under 8% for the best lever. Before
spending hours fetching more text, it is worth knowing what that spread is
made of — "add more books" and "add more books of a kind the model is bad at"
are different plans.

Three candidate explanations, all testable against the per-document
evaluation that shipped with the released model.

## 1. It is not length

If short documents scored worse simply because there is less of them, the
spread would be an artifact and the right action would be to ignore it.
Correlation between document size in bytes and bits per byte across the 71
books: **r = −0.122**. Weak, and in the direction that would make *longer*
books slightly harder. Length does not explain the spread.

## 2. It is form, and form explains 45% of the variance

Labels are hand-assigned from title and author, before looking at any score,
and are written out in full in
[`analyze-documents.py`](analyze-documents.py) so they can be disputed.

| form | books | bytes | mean bpb | vs narrative |
| --- | ---: | ---: | ---: | ---: |
| narrative prose | 64 | 2,723,851 | 1.5450 | — |
| non-fiction | 3 | 87,016 | 1.8429 | +19.3% |
| experimental prose | 2 | 96,981 | 1.9063 | +23.4% |
| verse | 2 | 94,280 | 1.9743 | +27.8% |

These four labels account for **45% of the variance** in per-document bits per
byte. For comparison, the entire batch-size bracket moved the aggregate by
0.98% and the entire data-scaling result by 7.30%.

The corpus is **91% narrative prose by bytes**, and it contains **no drama at
all** — no Shakespeare, no Ibsen, no Wilde. By bytes, verse holds 3.1%, experimental
prose holds 3.2% and non-fiction holds 2.9%.

## 3. Repetition of an author is worth 5%

9 of the 71 books' authors contribute more than one book. Their books score
**1.5388** against **1.6197** for books by an author who appears once — a
**5.0%** advantage for having been seen before in a different book.

This is the finding that turns the table above from a description into a
prediction. Verse and non-fiction are both *hard* and *rare*. If they were
intrinsically unpredictable, more of them would not help. But the
repeated-author effect shows this model does measurably better on text whose
style it has seen elsewhere, at a scale — 5% — comparable to the form
penalties themselves. The hard categories are most likely under-trained
rather than impossible.

## The experiment this sets up

**Claim:** raising the non-narrative share of the training corpus from 9% to
roughly a third will cut the verse and non-fiction penalties by more than it
costs narrative prose.

**Design**, reusing the machinery the clean-data experiment already built:

* a `gutenberg-mixed-broad` corpus adding verse, drama and non-fiction until
  narrative prose is about two thirds of the bytes
* a control arm sharing a byte-identical evaluation split and a shared
  vocabulary, via the existing `eval_only` and `--tokenizer-from` flags
* the released 9,600-step configuration on both arms, since batch, context,
  learning rate and model size are all now known to be at or near optimal
* report **per form**, not only in aggregate — the aggregate is a weighted
  average over the mixture, and this project has already been burned once by
  reading one as if it were a property of the model

**What would falsify it:** if the verse and non-fiction penalties survive a
threefold increase in their training mass, then those forms are intrinsically
harder for a 1.77M-parameter model at 256 tokens of context, and the corpus is
not the lever either. That would be a more interesting result than the
expected one, and it is the reason to run it rather than assume it.

## Artifacts

| file | what |
| --- | --- |
| `analyze-documents.py` | the analysis, including the hand-assigned form labels |
| `document-summary.json` | every number in this document |
| `eval-test-data71-full-9600.json` | the per-document scores it reads |
