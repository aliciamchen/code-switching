import random
import json

# sample for how data should be saved
tangramC_test = [
    {
        "target": "C",
        "convention": "crane kick",
        "game": "xxx",
        "rep": 0,
        "speakerThisRep": "alice",
        "convo": [{"player": "xxx", "text": "sample", "time": 2, "role": "speaker"}],
    }
]

# each dict is a conversation
# TODO: assign a player who is speaker to each rep number
# (don't use color specific labels - those can add later. maybe just use alice, bob, carol, etc)

# later - make big games by, for each game choose what is shared and what is unique
# this will determine the possible conventions for each target
# then for each target choose a convention, and for each convention choose a game (which determines the convo)
# NOTE; have to translate the arbitrary speaker labels to alice/bob etc. - each rep corresponds to a fixed speaker
# e.g. alice 0, bob 1, carol 2, dave 3, alice 4, bob 5
# each of these correspodns to an avatar

conventions = {
    "D": {
        "shared": ["priest"],
        "unique": ["wizard guy", "long sleeved book"],
    },  # etc
}
# if shared draw from shared, if unique draw one from unique
# TODO: should the unique ones be completely separate from the shared ones? like if in one item "preist" is shared, does that mean that the unique tangrams for another tangram cant include "priest"?


conventions_games = {"D": {"priest": ["xxx", "yyy"], "wizard guy": ["zzz"], "long sleeved book": ["qqq"]}}  # etc
# for "shared" find a list that has at least 2 elements

# tangrams = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
tangrams = ["D"]


def choose_tangrams(tangrams: list):
    """Choose a subset of 6 tangrams from the list of tangrams"""
    return random.sample(tangrams, 1)


def choose_shared_unique(tangrams: list):
    """Out of the 6 chosen tangrams, 3 should be 'shared' and 3 should be 'unique'"""
    shared = random.sample(tangrams, 0)
    unique = [t for t in tangrams if t not in shared]
    return {"shared": shared, "unique": unique}


def choose_labels(shared_unique: dict, conventions: dict):
    """For each tangram, select its labels for each group"""
    shared_tangrams = shared_unique["shared"]
    unique_tangrams = shared_unique["unique"]

    shared_labels = {}
    for shared_tangram in shared_tangrams:
        # for each tangram, randomly select a label in conventions[shared_tangram]["shared"]
        label = random.choice(
            conventions[shared_tangram]["shared"]
        )
        shared_labels[shared_tangram] = {}
        shared_labels[shared_tangram]["red"] = label
        shared_labels[shared_tangram]["blue"] = label

    unique_labels = {}
    for unique_tangram in unique_tangrams:
        unique_labels[unique_tangram] = {}
        unique_labels[unique_tangram]["red"] = random.choice(
            conventions[unique_tangram]["unique"]
        )
        remaining_labels = [
            label
            for label in conventions[unique_tangram]["unique"]
            if label != unique_labels[unique_tangram]["red"]
        ]
        unique_labels[unique_tangram]["blue"] = random.choice(remaining_labels)

    return {"shared_labels": shared_labels, "unique_labels": unique_labels}



def choose_games(labels: dict, conventions_games: dict):
    """Format of output should be {"red": {"A": {"game": "xxx", "label": "labelA1", "shared": "shared"}, "B": {"game": "yyy", "label": "labelB1", "shared": "unique"}}, "blue": {"A": {"game": "zzz", "label": "labelA2"}, "B": {"game": "www", "label": "labelB2", "shared": "unique"}}}"""
    groups = ['red', 'blue']
    shared_labels = labels["shared_labels"]
    unique_labels = labels["unique_labels"]

    output = {}
    output["red"] = {}
    output["blue"] = {}

    for tangram in shared_labels.keys():
        this_tangram_label = shared_labels[tangram]["red"] # can either be red or blue; they are the same
        possible_games = conventions_games[tangram][this_tangram_label]
        output["red"][tangram] = {}
        output["red"][tangram]["label"] = shared_labels[tangram]["red"]
        output["red"][tangram]["shared"] = "shared"
        output["red"][tangram]["game"] = random.choice(possible_games)

        output["blue"][tangram] = {}
        output["blue"][tangram]["label"] = shared_labels[tangram]["blue"]
        output["blue"][tangram]["shared"] = "shared"
        # choose a game for the blue game, which can't be the same as red game
        remaining_games = [game for game in possible_games if game != output["red"][tangram]["game"]]
        output["blue"][tangram]["game"] = random.choice(remaining_games)

    for tangram in unique_labels.keys():
        for group in groups:
            this_tangram_label = unique_labels[tangram][group]
            possible_games = conventions_games[tangram][this_tangram_label]
            output[group][tangram] = {}
            output[group][tangram]["label"] = this_tangram_label
            output[group][tangram]["shared"] = "unique"
            output[group][tangram]["game"] = random.choice(possible_games)

    # sort each of the tangrams alphabetically by group
    for group in groups:
        output[group] = dict(sorted(output[group].items()))

    return output


if __name__ == "__main__":
    chosen_tangrams = choose_tangrams(tangrams)
    shared_unique = choose_shared_unique(chosen_tangrams)
    print(shared_unique)
    labels = choose_labels(shared_unique, conventions)
    print(labels)
    games = choose_games(labels, conventions_games)
    print(games)

    # save as a json file
    with open("tangramC_test.json", "w") as f:
        json.dump(tangramC_test, f)
