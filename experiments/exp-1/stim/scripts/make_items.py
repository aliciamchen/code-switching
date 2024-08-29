"""
This script generates the items (e.g. what set of conversations should each participant see?)
"""

import json
import random
import pandas as pd

random.seed(88)

tangrams = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def choose_tangrams(tangrams: list):
    """Choose a subset of 6 tangrams from the list of tangrams"""
    # TODO: later counterbalance across participants
    return random.sample(tangrams, 6)


def choose_shared_unique(tangrams: list):
    """Out of the 6 chosen tangrams, 3 should be 'shared' and 3 should be 'unique'"""
    shared = random.sample(tangrams, 3)
    unique = [t for t in tangrams if t not in shared]
    return {"shared": shared, "unique": unique}


def choose_labels(shared_unique: dict, conventions: dict):
    """For each tangram, select its labels for each group"""
    shared_tangrams = shared_unique["shared"]
    unique_tangrams = shared_unique["unique"]

    shared_labels = {}
    for shared_tangram in shared_tangrams:
        # for each tangram, randomly select a label in conventions[shared_tangram]["shared"]
        label = random.choice(conventions[shared_tangram]["shared"])
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
    groups = ["red", "blue"]
    shared_labels = labels["shared_labels"]
    unique_labels = labels["unique_labels"]

    output = {}
    output["red"] = {}
    output["blue"] = {}

    for tangram in shared_labels.keys():
        this_tangram_label = shared_labels[tangram][
            "red"
        ]  # can either be red or blue; they are the same
        possible_games = conventions_games[tangram][this_tangram_label]
        output["red"][tangram] = {}
        output["red"][tangram]["label"] = shared_labels[tangram]["red"]
        output["red"][tangram]["shared"] = "shared"
        output["red"][tangram]["game"] = random.choice(possible_games)

        output["blue"][tangram] = {}
        output["blue"][tangram]["label"] = shared_labels[tangram]["blue"]
        output["blue"][tangram]["shared"] = "shared"
        # choose a game for the blue game, which can't be the same as red game
        remaining_games = [
            game for game in possible_games if game != output["red"][tangram]["game"]
        ]
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


def make_2AFC_lexicon(choose_games_output, chosen_tangrams):
    """Takes in the output of choose_games
    Make in the format of e.g.,
    [
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
        ]
    Chosen tangrams is available tangrams (e.g. 6)
    """
    lexicon = []
    for tangram in chosen_tangrams:
        for group in ["red", "blue"]:
            if choose_games_output[group][tangram]["shared"] == "shared":
                shared = "shared"
            else:
                shared = "unique"
            label = choose_games_output[group][tangram]["label"]
            lexicon.append(
                {"tangram": tangram, "shared": shared, "group": group, "label": label}
            )

    # order alphabetically by tangram
    lexicon = sorted(lexicon, key=lambda x: x["tangram"])
    return lexicon


def main(n_items):
    """wrapper to make n items"""
    # load in conventions_games.json
    with open("conventions_games.json", "r") as f:
        conventions_games = json.load(f)

    # store what can be shared and what can be unique
    conventions = {}
    for tangram, labels in conventions_games.items():
        conventions[tangram] = {"shared": [], "unique": []}
        for label, games in labels.items():
            if len(games) > 1:
                conventions[tangram]["shared"].append(label)
            else:
                conventions[tangram]["unique"].append(label)
    with open("conventions.json", "w") as f:
        json.dump(conventions, f)

    for i in range(n_items):
        chosen_tangrams = choose_tangrams(tangrams)
        shared_unique = choose_shared_unique(chosen_tangrams)
        labels = choose_labels(shared_unique, conventions)
        games = choose_games(labels, conventions_games)
        twoafc_lexicon = make_2AFC_lexicon(games, chosen_tangrams)

        # save
        with open(f"../items/item_{i}_lexicon.json", "w") as f:
            json.dump(twoafc_lexicon, f, indent=2)

        with open(f"../items/item_{i}_game_info.json", "w") as f:
            json.dump(games, f, indent=2)


if __name__ == "__main__":

    main(n_items=10)  # generate 10 independent sets of items
