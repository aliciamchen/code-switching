# Experiment 1

## `/stim`

This folder contains the stimuli and randomization for Experiment 1 (as well as the scripts needed to generate them)

### Game info from Boyce data

`conventions.json` contains all the possible final referring expressions, for each tangram. The `shared` expressions correspond to the expressions that two or more groups converged on, and the `unique` expressions are the (abstract) expressions that one group converged on. These expressions were selected by manually inspecting the final referring expressions (round 5) from `../boyce_data/filtered_chat.csv`, and filtering by games where 2/3 or 3/3 participants got the answer right.

`conventions_games.json` contains the `gameId` that each of the conventions in `conventions.json` belongs to, formatted this way for ease of access

`scripts/get_convos.py` extracts and formats the conversation and selection history for each tangram-game pair in `conventions_games.json`. It uses the chat data in `boyce_data/filtered_chat.csv` and the response data in `boyce_data/round_results.csv`; outputs are saved in `convos`. These files are among the inputs that are needed for generating the videos.

### Making the items, stimuli, trials, etc:

`items` contains the game info and lexicon for each counterbalancing assignment (called an 'item') here. (these names might have to be changed later if we actually use different items)

Right now, these are generated manually by choosing from the referring expressions in `conventions.json` and `conventions_games.json`. Which tangrams are shared and unique are arbitrary selected (and the assigment is reversed by counterbalancing assignment), and the games to pull from for each expression are chosen in order from `conventions_games.json`

The files in `items` are used to generate the `.json` files in `2AFC_trials` and to the video generation script. And both the files in `items` and `2AFC_trials` are also direct inputs to the javascript experiment.

### Making the individual stimuli

`make_videos.py` generates all the video stimuli for a specified item number. It loads in the game info for the specified item from `../items`, and finds the relevant conversation info from `../convos`. Videos are saved in `../convo_vids/videos`.

`make_2afc_trials.py` takes each item and generates its corresponding 2AFC trials. Right now for the 'unseen' tangrams, are taken from the other counterbalancing assignment

## scp ing to MIT scripts

`scp -r exp-1 aliciach@athena.dialup.mit.edu:~/www/tangrams/v1`

`rsync -av --exclude 'stim/boyce_data' --exclude 'stim/convo_vids/videos/480p15/partial_movie_files' --exclude 'stim/convos' --exclude 'stim/convo_vids/texts' exp-1/ aliciach@athena.dialup.mit.edu:~/www/tangrams`


website is at https://web.mit.edu/aliciach/www/tangrams/v1/exp-1/

https://web.mit.edu/aliciach/www/tangrams