# Earlier vs. later (Experiment 3)

This experiment tests whether participants choose an earlier (longer, more descriptive) or later (shorter, converged) label for the same tangram, depending on their audience and goal. This folder contains the jsPsych experiment code, and `stim/` contains the stimuli and randomization along with the scripts that generate them. The stimulus pipeline mirrors the shared-unique experiment (Experiment 2), which shares the same set of observation videos.

## Game info from the Boyce et al. data

The stimuli are built from a corpus of online multiplayer reference games (Boyce et al.), included here as `stim/boyce_data/filtered_chat.csv` (chat messages) and `stim/boyce_data/round_results.csv` (listener responses).

`stim/scripts/conventions.json` contains the candidate final referring expressions for each tangram. The `shared` expressions are those that two or more groups converged on, and the `unique` expressions are the (abstract) expressions that a single group converged on. These expressions were selected by manually inspecting the final referring expressions (round 5) in `filtered_chat.csv`, keeping games where 2/3 or 3/3 listeners chose the correct tangram.

`stim/scripts/conventions_games.json` records the `gameId` that each expression in `conventions.json` came from, formatted for ease of access.

`stim/scripts/get_convos.py` extracts and formats the conversation and selection history for each tangram–game pair in `conventions_games.json`, using the chat and response data above. The outputs are saved in `stim/convos` and are inputs to the video generation step.

## Items and trials

`stim/items` contains the game info and lexicon for each counterbalancing assignment (called an "item"). These files were assembled by choosing referring expressions from `conventions.json`; which tangrams are shared versus unique is arbitrary (and reversed across counterbalancing assignments), and the source games for each expression are taken in order from `conventions_games.json`.

The files in `stim/items` generate the `.json` files in `stim/2AFC_trials`, and both directories are direct inputs to the JavaScript experiment.

## Generating the stimuli

`stim/scripts/make_videos.py` generates the video stimuli for a specified item. It loads the game info for that item from `stim/items`, finds the corresponding conversations in `stim/convos`, and saves the videos to `stim/convo_vids/videos`. The rendered videos are not committed to the repository because of their size; they are available on OSF at https://osf.io/5j6uk/files/osfstorage.

`stim/scripts/make_2afc_trials.py` generates each item's 2AFC trials. The "unseen" tangram labels (used in manipulation-check trials) are taken from the other counterbalancing assignment.
