import itertools
import random
import json

random.seed(88)
# TODO: counterbalancing of 2AFC choices
items = [
    {
        "item_id": 1,
        "lexicon": [
            {"tangram": "A", "shared": "unique", "group": "blue", "label": "labelA1"},
            {"tangram": "A", "shared": "unique", "group": "red", "label": "labelA2"},
            {"tangram": "C", "shared": "unique", "group": "blue", "label": "labelC1"},
            {"tangram": "C", "shared": "unique", "group": "red", "label": "labelC2"},
            {"tangram": "F", "shared": "shared", "group": "both", "label": "labelF"},
            {"tangram": "H", "shared": "shared", "group": "both", "label": "labelH"},
            {"tangram": "I", "shared": "shared", "group": "both", "label": "labelI"},
            {"tangram": "J", "shared": "unique", "group": "blue", "label": "labelJ1"},
            {"tangram": "J", "shared": "unique", "group": "red", "label": "labelJ2"},
        ],
    }
]


def separate_tangrams(lexicon):
    shared_tangrams = [entry for entry in lexicon if entry["shared"] == "shared"]
    unique_tangrams = [entry for entry in lexicon if entry["shared"] == "unique"]
    return shared_tangrams, unique_tangrams


# Generate trial sequences for each participant
# Sample trial: {"goal": "refer", "shared": {"group": "both", "tangram": "D", "label": "labelD"}, "unique": {"group": "blue", "tangram": "A", "label": "labelA1"}}
# On each trial, participants should see one shared tangram and one unique tangram.
# Over an experiment, participants should see each shared tangram twice and each unique tangram once, so that they observe all the tangram-label pairs.
# Half of the trials should have the "refer" goal and half of the trials should have the "social" goal


def generate_all_unique_sets(shared_tangrams, unique_tangrams):
    """Generate all unique sets of trials"""

    # what are the tangrams in 'unique'?
    unique_set = list(set([entry["tangram"] for entry in unique_tangrams]))
    shared_set = list(set([entry["tangram"] for entry in shared_tangrams]))

    all_sets = []
    shared_permutations = list(itertools.permutations(shared_set))
    print(len(shared_permutations))

    # TODO: fix: doesn't have all combinations of refer and social
    # instad, it makes the red group ones "refer" and the blue group ones "social"
    # later: how to fix?

    refer_social_pairs = [["refer", "social"], ["social", "refer"]]

    # Generate all combinations of the valid pairs
    refer_social_combs = list(itertools.product(refer_social_pairs, repeat=3))

    # for red_goal in ["refer", "social"]:
    for comb in refer_social_combs:
        for shared_perm in shared_permutations:
            trials = []
            for i in range(3):
                # goals = ["refer", "social"]
                trial_red = {
                    "goal": comb[i][0],
                    "shared": [
                        entry
                        for entry in lexicon
                        if (entry["group"] == "red")
                        and (entry["tangram"] == shared_perm[i])
                    ][0],
                    "unique": [
                        entry
                        for entry in lexicon
                        if (entry["group"] == "red")
                        and (entry["tangram"] == unique_set[i])
                    ][0],
                }
                trial_blue = {
                    "goal": comb[i][1],
                    "shared": [
                        entry
                        for entry in lexicon
                        if (entry["group"] == "red")
                        and (entry["tangram"] == shared_perm[i])
                    ][0],
                    "unique": [
                        entry
                        for entry in lexicon
                        if (entry["group"] == "blue")
                        and (entry["tangram"] == unique_set[i])
                    ][0],
                }
                trials.append(trial_red)
                trials.append(trial_blue)
            all_sets.append(trials)

    # # Ensure each shared tangram appears twice
    # shared_tangrams = shared_tangrams * 2

    # # Generate all permutations of shared tangrams and unique tangrams
    # unique_permutations = list(itertools.permutations(unique_tangrams))

    # # Generate all possible combinations of 3 "refer" and 3 "social" goals
    # goals = ["refer"] * 3 + ["social"] * 3
    # goal_permutations = set(itertools.permutations(goals))
    # print(len(goal_permutations))
    # # print(shared_tangrams)

    # for unique_perm in unique_permutations:
    #     for goal_perm in goal_permutations:
    #         trials = []
    #         for i in range(6):
    #             trial = {
    #                 "goal": goal_perm[i],
    #                 "shared": shared_tangrams[i],
    #                 "unique": unique_perm[i],
    #             }
    #             trials.append(trial)
    #         all_sets.append(trials)

    return all_sets


# def generate_all_unique_sets(shared_tangrams, unique_tangrams):
#     """Generate all unique sets of trials, for a single item"""
#     goals = ["refer", "social"]
#     trials = []
#     # Each shared tangram should appear twice
#     shared_tangrams = shared_tangrams * 2

#     random.shuffle(shared_tangrams)
#     random.shuffle(unique_tangrams)

#     assert len(shared_tangrams) == len(unique_tangrams)

#     for i in range(len(shared_tangrams)):
#         goal = goals[i % len(goals)]
#         trial = {
#             "goal": goal,
#             "shared": shared_tangrams[i],
#             "unique": unique_tangrams[i],
#         }
#         trials.append(trial)

#     return trials


# def generate_trials_for_participants(num_participants, items):
#     """Generate trials for multiple participants, for each item"""
#     all_trials = []
#     for item in items:
#         lexicon = item["lexicon"]
#         shared_tangrams = [entry for entry in lexicon if entry["shared"] == "shared"]
#         unique_tangrams = [entry for entry in lexicon if entry["shared"] == "unique"]

#         trial_sets = []
#         for i in range(num_participants):
#             trials = generate_trials(shared_tangrams, unique_tangrams)
#             trial_sets.append(trials)

#         all_trials.append({"item_id": item["item_id"], "trials": trial_sets})

#     return all_trials


# all_trials = generate_trials_for_participants(num_participants=10, items=items)

with open("../items/item_0_lexicon.json", "r") as f:
    lexicon = json.load(f)

shared_tangrams, unique_tangrams = separate_tangrams(lexicon)
all_sets = generate_all_unique_sets(shared_tangrams, unique_tangrams)
print(len(all_sets))
with open("all_sets_test.json", "w") as json_file:
    json.dump(all_sets, json_file, indent=2)

print("Trials saved to trials.json")
