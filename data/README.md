# Data dictionary

This directory contains the preprocessed, de-identified data for all experiments. Each file is the output of the corresponding preprocessing script (`analysis/{experiment}/preprocess.py`), which reads the raw data, removes identifying information, and replaces platform IDs with random `subject_id` values. The raw data are not included in the repository because they are not anonymized. A few files (noted below) are instead written by the analysis scripts as intermediate outputs that downstream steps read.

Conventions shared across experiments:

- `subject_id` is a randomly generated identifier and cannot be linked back to a participant.
- Tangrams are identified by letters (A, B, C, ...).
- `item_id` (or `tangram_set` in the transparency task) identifies which of the three tangram sets the participant saw (0, 1, or 2), and `counterbalance` (a or b) identifies the assignment of tangrams and labels to the red versus blue group.
- `goal` is the assigned communicative goal on that trial: `refer` (help the audience identify the tangram) or `social` (be identified as a group member).
- `understood` is the participant's exit-survey response to "Did you understand the instructions?" (yes/no); participants who answered "no" were excluded from analysis.

## Exit surveys

Every experiment folder contains an `exit_survey.csv` with one row per participant:

| Column | Description |
| --- | --- |
| `subject_id` | Anonymized participant identifier |
| `understood` | Whether the participant understood the instructions (`yes`/`no`) |
| `age` | Age in years (free-response text box) |
| `gender` | Selected from the options `male`, `female`, `nonconforming` (shown as "Gender Variant/Non-Conforming"), or `abstain` (shown as "Prefer not to answer") |
| `comments` | Optional free-response comments |

## `free-response/` (Experiment 1)

**`selection_trials.csv`** — one row per selection-phase trial. Participants chose a tangram and typed a description for it.

| Column | Description |
| --- | --- |
| `trial_num` | Trial index within the session |
| `n_blue` | Number of blue-group members in the audience (0–4) |
| `n_naive` | Number of naive people in the audience (0, 1, 2, 4, 8, or 16) |
| `goal` | Assigned goal (`refer`/`social`) |
| `previous_selection` | Tangram chosen on the immediately preceding trial (blocked from being selected again) |
| `selected_tangram` | Tangram the participant chose to describe |
| `selected_tangram_group` | Whether the chosen tangram was discussed only by the blue group (`blue_specific`), only by the red group (`red_specific`), or by both (`shared`) |
| `selected_tangram_earlier_red`, `selected_tangram_earlier_blue` | Each group's earlier (initial-block) label for the chosen tangram, where one exists |
| `selected_tangram_later_red`, `selected_tangram_later_blue` | Each group's later (converged) label for the chosen tangram, where one exists |
| `written_label` | The description the participant typed |
| `sbert_cosine_earlier_red`, `sbert_cosine_earlier_blue`, `sbert_cosine_later_red`, `sbert_cosine_later_blue` | Cosine similarity between sentence-embedding vectors (`paraphrase-MiniLM-L6-v2`) of `written_label` and each reference label |
| `utt_length` | Number of words in `written_label` |

**`selection_trials_filtered.csv`** — the same trials (plus a precomputed `prop_naive` column), restricted to participants above the median on the utterance-memory learning metric computed in the analysis script. This file is written by `free-response_analysis.Rmd` and feeds the appendix figure for high-learning participants.

**`converged_expressions.csv`** — a lookup table of each group's converged later label for every tangram in every item/counterbalance assignment, used to score similarity between participants' descriptions and the observed conventions.

## `shared-unique/` (Experiment 2)

**`selection_trials.csv`** — one row per selection-phase trial. Participants chose between two tangram–label pairs.

| Column | Description |
| --- | --- |
| `type` | `diff` for critical trials (two different tangrams, one with a group-specific label and one with a shared label); `same` for manipulation-check trials (two candidate labels for the same tangram) |
| `goal` | Assigned goal (`refer`/`social`) |
| `audience` | Whether the audience was a specified group (`one`) or could be from `either` group |
| `audience_group` | The specified audience group (`red`/`blue`) |
| `option1.*`, `option2.*` | The two response options: `.tangram` (tangram letter), `.shared` (`shared` or `unique` label type), `.group` (which group's label is shown), `.label` (label text) |
| `response.*` | The same fields for the option the participant chose |

**`selection_trials_clean.csv`** — critical trials recoded for analysis: `condition` combines goal and audience (`refer one`, `refer either`, `social one`), `shared.tangram` and `unique.tangram` identify which tangram carried each label type, and `response.unique` records whether the participant chose the group-specific (`unique`) or `shared` option.

## `earlier-later/` (Experiment 3)

**`selection_trials.csv`** — one row per selection-phase trial. Participants chose between an earlier and a later label for the same tangram.

| Column | Description |
| --- | --- |
| `type` | `main` for critical trials; `baseline` for manipulation-check trials |
| `goal`, `audience`, `audience_group` | As in Experiment 2 |
| `option1.*`, `option2.*` | The two labels: `.tangram`, `.shared` (whether the tangram had a `shared` or group-`unique` label), `.earlier` (`earlier`/`later`), `.length` (label length in words), `.group`, `.label` (label text) |
| `response.*` | The same fields for the chosen label |

**`selection_trials_clean.csv`** — critical trials recoded for analysis, with `condition` (`refer one`, `refer either`, `social one`) and `response.earlier` recording whether the participant chose the `earlier` or `later` label.

**`labels.csv`** — one row per 2AFC label pair (item × counterbalance × tangram × audience group), with the `earlier_label` and `later_label` texts and `n`, the number of critical-trial observations for that pair. This file is written by `earlier-later_analysis.Rmd` and is an input to the computational model fitting.

## `transparency/` (naive observer task)

**`selection_trials.csv`** — one row per guess. Naive participants saw a label and guessed which tangram it referred to.

| Column | Description |
| --- | --- |
| `tangram_set` | Which of the three tangram sets the label came from (0–2) |
| `target` | The correct tangram |
| `earlier` | Whether the label shown was an `earlier` or `later` label |
| `length` | Label length in words |
| `label` | The label text shown |
| `response` | The tangram the participant guessed |
| `correct` | Whether the guess matched the target (`True`/`False`) |

**`means_by_label.csv`** — bootstrapped guessing accuracy for each individual label (`n` guesses; `empirical_stat` is the observed proportion correct; `mean`, `ci_lower`, and `ci_upper` are the bootstrap mean and 95% confidence interval). Written by `transparency_analysis.Rmd`; the Experiment 4 analysis and model read these item-level transparency estimates.

**`means_agg.csv`** — the same bootstrap summary aggregated over all earlier versus all later labels.

## `varied_audience/` (Experiment 4)

**`selection_trials.csv`** — one row per selection-phase trial. Participants chose between an earlier and a later label for a tangram, for audiences of varying composition.

| Column | Description |
| --- | --- |
| `type` | `main` for critical trials; `baseline` for manipulation-check trials |
| `goal` | Assigned goal (`refer`/`social`) |
| `n_ingroup` | Number of blue-group members in the audience (0–4) |
| `n_outgroup` | Number of naive people in the audience (0, 1, 2, 4, 8, or 16) |
| `option1.*`, `option2.*` | The two labels: `.tangram`, `.earlier` (`earlier`/`later`), `.length` (label length in words), `.label` (label text) |
| `response.*` | The same fields for the chosen label |

**`selection_trials_clean.csv`** — critical trials recoded for analysis, keeping the audience composition (`n_ingroup`, `n_outgroup`), `goal`, `tangram`, and `response.earlier`.
