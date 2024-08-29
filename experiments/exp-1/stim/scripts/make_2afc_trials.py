import random
import json
# TODO: counterbalancing of 2AFC choices
items = [
    {
        "item_id": 1,
        "lexicon": [
            {"tangram": "A", "shared": "unique", "group": "blue", "label": "labelA1"},
            {"tangram": "A", "shared": "unique", "group": "red", "label": "labelA2"},
            {"tangram": "B", "shared": "unique", "group": "blue", "label": "labelB1"},
            {"tangram": "B", "shared": "unique", "group": "red", "label": "labelB2"},
            {"tangram": "C", "shared": "unique", "group": "blue", "label": "labelC1"},
            {"tangram": "C", "shared": "unique", "group": "red", "label": "labelC2"},
            {"tangram": "D", "shared": "shared", "group": "both", "label": "labelD"},
            {"tangram": "E", "shared": "shared", "group": "both", "label": "labelE"},
            {"tangram": "F", "shared": "shared", "group": "both", "label": "labelF"},
            {"tangram": "G", "shared": "shared", "group": "both", "label": "labelG"},
            {"tangram": "H", "shared": "shared", "group": "both", "label": "labelH"},
            {"tangram": "I", "shared": "shared", "group": "both", "label": "labelI"},
            {"tangram": "J", "shared": "unique", "group": "blue", "label": "labelJ1"},
            {"tangram": "J", "shared": "unique", "group": "red", "label": "labelJ2"},
            {"tangram": "K", "shared": "unique", "group": "blue", "label": "labelK1"},
            {"tangram": "K", "shared": "unique", "group": "red", "label": "labelK2"},
            {"tangram": "L", "shared": "unique", "group": "blue", "label": "labelL1"},
            {"tangram": "L", "shared": "unique", "group": "red", "label": "labelL2"},
        ],
    }
]


# Generate trial sequences for each participant
# Sample trial: {"goal": "refer", "shared": {"group": "both", "tangram": "D", "label": "labelD"}, "unique": {"group": "blue", "tangram": "A", "label": "labelA1"}}
# On each trial, participants should see one shared tangram and one unique tangram.
# Over an experiment, participants should see each shared tangram twice and each unique tangram once, so that they observe all the tangram-label pairs.
# Half of the trials should have the "refer" goal and half of the trials should have the "social" goal


def generate_trials(shared_tangrams, unique_tangrams):
    """Generate a set of trials, for one participant"""
    goals = ["refer", "social"]
    trials = []
    # Each shared tangram should appear twice
    shared_tangrams = shared_tangrams * 2

    random.shuffle(shared_tangrams)
    random.shuffle(unique_tangrams)

    assert len(shared_tangrams) == len(unique_tangrams)

    for i in range(len(shared_tangrams)):
        goal = goals[i % len(goals)]
        trial = {
            "goal": goal,
            "shared": shared_tangrams[i],
            "unique": unique_tangrams[i],
        }
        trials.append(trial)

    return trials


def generate_trials_for_participants(num_participants, items):
    """Generate trials for multiple participants, for each item"""
    all_trials = []
    for item in items:
        lexicon = item["lexicon"]
        shared_tangrams = [entry for entry in lexicon if entry["shared"] == "shared"]
        unique_tangrams = [entry for entry in lexicon if entry["shared"] == "unique"]

        trial_sets = []
        for i in range(num_participants):
            trials = generate_trials(shared_tangrams, unique_tangrams)
            trial_sets.append(trials)

        all_trials.append({"item_id": item["item_id"], "trials": trial_sets})

    return all_trials


all_trials = generate_trials_for_participants(num_participants=10, items=items)

with open("trials.json", "w") as json_file:
    json.dump(all_trials, json_file, indent=2)

print("Trials saved to trials.json")
