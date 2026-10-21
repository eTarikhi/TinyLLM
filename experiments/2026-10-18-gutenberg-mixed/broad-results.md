# Broadening the corpus by form: the mixture is free, the volume is not

`corpus-composition.md` measured what the 71 test books disagree about and
specified this experiment. Form explains 45% of the per-document spread —
verse cost +27.8% against narrative prose, experimental prose +23.4%,
non-fiction +19.3% — while those three forms held 9.3% of the training
bytes and drama was absent entirely. The claim under test: **those penalties
are under-training, not intrinsic difficulty**, so adding drama, verse and
non-fiction should close them.

## Design

Three arms, 4,800 steps each, seed 1337, identical schedule, identical
evaluation grid, byte-identical `val.bin`, `test.bin` and `tokenizer.json`
(verified with `cmp`). The additions are `train_only` everywhere, so all
three arms are scored on the same 71 books.

| arm | train tokens | narrative share | what was added |
| --- | ---: | ---: | --- |
| `broad-ctrl` | 18,939,699 | 90.7% | nothing |
| `broad-narrative` | 26,057,637 | ~93% | 20.5 MB narrative prose |
| `broad-full` | 25,952,894 | 67.8% | 20.5 MB drama, verse, non-fiction |

The two additions match to 0.02% in bytes, which is what makes the
decomposition below possible. `broad-ctrl` reproduces the published
`data71-full` test figure to four decimal places — **1.6147** — from a corpus
prepared independently, which is the strongest end-to-end reproducibility
check this project has.

## The aggregate is a trade, not a win

| arm | test bpb | vs ctrl |
| --- | ---: | ---: |
| `broad-ctrl` | 1.6147 | — |
| `broad-narrative` | 1.6297 | +0.93% |
| `broad-full` | 1.6338 | +1.19% |

Both additions make the aggregate worse. That number is 91% narrative prose
by mixture, so it mostly reports what happened to narrative prose. Per form:

| form | n | ctrl | narrative | full | full vs ctrl | full vs narrative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| narrative | 64 | 1.5913 | 1.6027 | 1.6143 | +1.44% | +0.72% |
| experimental | 2 | 1.9427 | 1.9612 | 1.9684 | +1.32% | +0.37% |
| non-fiction | 3 | 1.9124 | 1.9527 | 1.8861 | −1.38% | −3.41% |
| verse | 2 | 2.0268 | 2.0161 | 1.9493 | −3.82% | −3.31% |

Every entry in the last two columns except experimental is several times
the 0.37% seed bar. The forms are small — two books of verse, three of
non-fiction — but each moves consistently against *both* comparisons.

## Two findings, and the second is the new one

**1. Targeting a form with training mass reduces that form's penalty.**
Measured against narrative prose within each arm, the verse penalty fell
from +27.4% to +20.8% and non-fiction from +20.2% to +16.8%. Experimental
prose, the one form nothing was added for, did not move (−0.1 percentage
points) — an internal control that came free with the design. That is the
under-trained reading confirmed, and it is not "more data helps everything":
the volume-matched narrative arm left verse almost where the control did
(2.0161 against 2.0268).

**2. Volume itself is the cost, and it costs the same whatever the form.**
Adding 37% more *narrative prose* — the dominant form, the thing the model
is already best at — cost +0.93%. Swapping that addition for drama, verse
and non-fiction cost a further **+0.25%, inside the seed bar**. In other
words: on a fixed evaluation set, at 1.77M parameters, extra text competes
for capacity and the competition is indiscriminate. The mixture change is
essentially free in aggregate while buying verse −3.31% and non-fiction
−3.41% against the volume-matched arm, paid for by narrative +0.72%.

Finding 2 explains a number this project has seen twice without isolating
it. `data-gb71` cost +0.53% when 44 books were added to training against a
fixed test split, and was read then as "those books are a different
distribution". The narrative arm shows the effect needs no distribution
story: same form, same period, same register still costs. Capacity is what
the text competes for.

## The 2×2: capacity halves the cost, it does not remove it

`broad-4L256d-{ctrl,full}` repeat the comparison at 3,737,856 parameters,
clock-matched at 2,571 steps. The control again reproduces a previously
published figure exactly — **1.6365**, the `size-4L256d-gb71` number — so
both cells sit on measurements this project already trusts.

| | broad-ctrl | broad-full | gap |
| --- | ---: | ---: | ---: |
| 1.77M | 1.6147 | 1.6338 | +1.19% |
| 3.74M | 1.6365 | 1.6467 | +0.62% |

Doubling the parameters roughly **halves** the cost of broadening. Per form
the same trade repeats at the larger size — verse 2.0415 → 1.9701 (−3.50%),
non-fiction 1.9492 → 1.9046 (−2.29%), narrative 1.6132 → 1.6260 (+0.79%),
experimental 1.9592 → 1.9680 (+0.45%) — and the verse penalty closes by the
same ~6 percentage points at both sizes (27.4% → 20.8% at 1.77M, 26.5% →
21.2% at 3.74M). What capacity buys is not a larger form gain; it is a
smaller narrative loss (+1.44% → +0.79%). That is what "the text competes
for capacity" predicts: the benefit of targeting a form is roughly
size-independent, and the price falls as the model grows.

Extrapolating the halving is tempting and should be resisted: two points do
not give a rate. What the 2×2 establishes is direction — capacity is *a*
constraint, not necessarily the only one.

## The released horizon: length trims it too

`broad-full-9600` runs the released configuration on the broad corpus.

| | 4,800 steps | 9,600 steps |
| --- | ---: | ---: |
| 71 books | 1.6147 | 1.5662 |
| broad | 1.6338 | 1.5790 |
| cost of broadening | +1.19% | +0.82% |

Per form against the released model: verse 1.9743 → 1.8996 (−3.78%),
non-fiction 1.8429 → 1.8173 (−1.39%), narrative 1.5450 → 1.5604 (+1.00%),
experimental 1.9063 → 1.9166 (+0.54%). The shape never changes; only the
price does. Horizon and capacity both behave like slack in the same budget:
each reduces what the added text costs the dominant form, and neither
changes how much the targeted forms gain.

## The long clock completes the grid: the cost closes

`broad-4L256d-{ctrl,full}-long` (5,173 steps, clock-matched to the
9,600-step small runs) complete the size × horizon grid:

| cost of broadening | 4,800-step clock | 9,600-step clock |
| --- | ---: | ---: |
| 1.77M parameters | +1.19% | +0.82% |
| 3.74M parameters | +0.62% | **+0.28%** |

0.28% is inside the 0.37% seed bar. At 3.74M parameters on the long clock,
broadening the corpus from 91% to 68% narrative prose **costs nothing
measurable in aggregate** (ctrl 1.5544, full 1.5587), and both arms beat
the released 1.77M model (1.5662). Per form the trade is unchanged in
shape: non-fiction 1.8627 → 1.7791 (−4.49%), verse 1.9538 → 1.8677
(−4.41%), narrative 1.5319 → 1.5412 (+0.61%), experimental 1.8929 → 1.8905
(−0.13%). The form gains are roughly constant at every size and horizon;
what slack buys is the narrative price falling toward zero. Capacity and
horizon are, as suspected, the same budget spent two ways.

## Transfer to unseen books

Every number above was measured on books both arms trained on. The
held-out probe asks the other question: 40 further titles — ten each of
drama, verse, non-fiction and narrative prose, verified disjoint from
every training corpus (`gutenberg-heldout-forms`) — scored per document
for both long-clock checkpoints. Thirty-eight scored (two short verse
volumes fall under the per-document minimum).

| form | n | ctrl | full | delta |
| --- | ---: | ---: | ---: | ---: |
| narrative | 10 | 1.8257 | 1.7578 | −3.72% |
| drama | 10 | 2.3041 | 1.9876 | −13.73% |
| non-fiction | 10 | 1.7372 | 1.6792 | −3.34% |
| verse | 8 | 2.1569 | 1.9870 | −7.88% |
| **aggregate** | 38 | 1.9887 | 1.8271 | **−8.13%** |

On unseen books the broad model is better everywhere — dramatically at
drama (−13.73%), the form the control has literally never seen in
training. Against the narrative control group (−3.72%, the gain from
more and more-varied training text on books of a kind both arms know),
drama transfers an extra −10.01 percentage points and verse −4.16; held-out
non-fiction (+0.38 points beyond narrative) gains only what narrative
gains, so its improvement reads as volume rather than form targeting.

The asymmetry is the result to carry forward: **on seen books broadening
is now free; on unseen books it is worth 8%.** A fixed evaluation set
made broadening look like a trade. It is, on that set — and it is also
the only intervention in this project that buys generality.

## Replication at seed 1338

The original three arms were single-seed. Rerunning the pair at seed 1338
(ctrl 1.6216, full 1.6281) puts the aggregate cost at **+0.40%** against
seed 1337's +1.19% — same sign, a third of the size, and a reminder that
the aggregate is the small difference of two large opposing effects and
inherits their seed noise. The per-form pattern is stable across seeds:
narrative +0.54% (seed 1337: +1.44%), verse −3.38% (−3.82%), non-fiction
−2.64% (−1.38%), experimental −0.05% (+1.32%; two books, and the one form
nothing targets). Read the headline as "at 1.77M on the short clock,
broadening costs roughly half a percent to one percent in aggregate while
buying 1.5–4% on the forms it targets", not as the single-seed +1.19%.

## What it does not settle

The form labels for the held-out corpus come from search keywords, not
hand assignment, and its non-fiction group is where the two labelings are
hardest to compare (the original 71's non-fiction is Darwin, Douglass and
Thoreau; the probe's skews to essays and lectures). Whether an even
broader mixture (narrative at ~50%) keeps the aggregate at zero while
buying more transfer is untested. And all of this remains at 256 tokens
of context; nothing here says how the trade scales past 4M parameters.

## Artifacts

| file | what |
| --- | --- |
| `analyze-broad.py` | this analysis, from the committed eval JSON |
| `broad-summary.json` | every number above |
| `eval-test-broad-{ctrl,narrative,full}.json` | per-document test scores, 71 books each |
| `broad-selection.json`, `narrative-selection.json` | the two additions, resolved and verified |
| `tests/test_broad_corpus.py` | pins the mixture, the split policy, and the eval_only regression |
