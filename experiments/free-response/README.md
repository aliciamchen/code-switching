# Free-response (Experiment 1)

This experiment tests code-switching behavior in a free-response format: participants observe two groups playing a reference game, then choose a tangram and write their own description for it, given an audience and a goal. This folder contains the jsPsych experiment code, and `stim/` contains the stimuli and generation scripts. This experiment uses its own set of observation videos, separate from the set shared by Experiments 2--4; the rendered videos are not committed to the repository because of their size and are available on OSF at https://osf.io/5j6uk/files/osfstorage.

## Generating videos

Note that manim has a bug where it caches images improperly, causing some of the avatars to flicker within the same video. To regenerate the affected videos, run:

```
python stim/scripts/regenerate_problematic_files.py
```
