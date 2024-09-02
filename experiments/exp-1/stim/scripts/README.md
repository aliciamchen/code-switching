# Scripts to generate stimuli

This folder contains the scripts to generate the stimuli and randomization for Experiment 1.

## Files

### Game info from Boyce data

`conventions.json` contains all the possible final referring expressions, for each tangram. The `shared` expressions correspond to the expressions that two or more groups converged on, and the `unique` expressions are the (abstract) expressions that one group converged on. These expressions were selected by manually inspecting the final referring expressions (round 5) from `../boyce_data/filtered_chat.csv`, and filtering by games where 2/3 or 3/3 participants got the answer right.

`conventions_games.json` contains the `gameId` that each of the conventions in `conventions.json` belongs to, formatted this way for ease of access

`get_convos.py` extracts and formats the conversation and selection history for each tangram-game pair in `conventions_games.json`. It uses the chat data in `../boyce_data/filtered_chat.csv` and the response data in `../boyce_data/round_results.csv`; outputs are saved in `../convos`. These files are among the inputs that are needed for generating the videos.

### Making the items

`make_items.py` takes in `conventions.json`, the number of items to generate, and generates the high-level info for each item (i.e. which tangrams should appear in each item? which tangrams have shared labels, and which have unique labels? what labels should each of the groups see, for each tangram? for each tangram-label pair, which game should the conversation be drawn from) For each item, the tangrams, and the labels, are drawn without replacement from their respective available sets. The outputs are saved in `../stim/items`, labeled by their item number. These outputs are the direct inputs to the javascript experiment.

### Making the individual stimuli

`make_videos.py` generates all the video stimuli for a specified item number. It loads in the game info for the specified item from `../items`, and finds the relevant conversation info from `../convos`. Videos are saved in `../convo_vids/videos`.

`make_2afc_trials.py` takes each item and the specified number of items to loop over, and generates all the combinations of 2AFC trials for each item. They are generated so that each pari consists of one 'shared' tangram and one 'unique' tangram. There are twice as many 'unique' tangrams as 'shared' tangram, so each 'shared' tangram is paired with the same corresponding 'unique' tangram for each group. Then for each shared-unique pairing, the audience is either a specific group or any group (so two pairs are specific and one pair is any, or vice versa). There is a within-participant manipulation of goal, so each trial is seen with the "refer" goal once and with the "social" goal once. This assignment is all counterbalanced across participants, so for 6 tangrams (12 trials) there are 48 unique sets of trials. Each set of sets of trials is stored in `../2AFC_trials` by its item number.
