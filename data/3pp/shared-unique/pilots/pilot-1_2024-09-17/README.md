# Codebook

This version of the experiment was made with commit #5264c4d

Total 8 participants; 48 total trials

Audience manipulations were `red` or `blue` groups (no `both` groups for this pilot)

## `selection_trials.csv`: the 2AFC trials

- `subject_id`: anonymized participant ID
- `item_id`: which counterbalancing assignment of shared tangrams vs. unique tangrams? `0`, or `1` (the shared tangrams in `0` are the unique tangrams in `1`, and vice versa)
- `type`: are the two tangrams in the trial `diff`ererent or are they the `same`?
- `goal`: `social` (identify as member of own group) or `refer` (pick out the correct tangram)
- `audience`: which group (`blue` or `red`) saw the utterance
- 2AFC options (`option1` and `option2`) and participant response (`response`)
  - `tangram`: which tangram
  - `shared`: was the label `shared` between the two groups, or was it `unique` to one group?
  - `group`: which group (`red` or `blue`) did the tangram-label pair belong to, or was it an `unseen` label (i.e was it one of the unique tangrams that was in the other counterbalancing assignment)
  - `label`: the displayed label for the tangram
- `understood`: did they say in the exit survey that they understood the instructions?

## `exit_survey.csv`

- `subject_id`: anonymized participant ID
- `understood`: did they understand the instructions?
- `gender`: `male`, `female`, or `nonconforming`
- `feedback`: feedback from the participants so that we can improve this task in the future
- `comments`: any other comments from participants
