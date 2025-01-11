import itertools
import random
import json

random.seed(88)

ns_ingroup = [0, 1, 2, 3, 4]
ns_outgroup = [0, 1, 2, 4, 8, 16]
ns = [[i, o] for i in ns_ingroup for o in ns_outgroup if (i, o) != (0, 0)]

goals = ["refer", "social"]


def separate_tangrams(lexicon):
    shared_tangrams = [entry for entry in lexicon if entry["shared"] == "shared"]
    unique_tangrams = [entry for entry in lexicon if entry["shared"] == "unique"]
    return shared_tangrams, unique_tangrams


def generate_main_trials(lexicon):

    tangrams = sorted(list(set([entry["tangram"] for entry in lexicon])))

    trials = []
    for tangram in tangrams:
        for n in ns:
            for goal in goals:
                options = [
                    entry
                    for entry in lexicon
                    if (entry["group"] == "blue") and (entry["tangram"] == tangram)
                ]
                if len(options) == 2:
                    trial = {
                        "type": "main",  # main or control/baseline
                        "goal": goal,
                        "tangram": tangram,
                        "n_ingroup": n[0],
                        "n_outgroup": n[1],
                        "options": options,
                    }
                    trials.append(trial)

    return trials


def generate_control_trials(this_lexicon, other_lexicon):
    """other_lexicon is lexicon of another (counterbalanced) items to get the labels that are not seen"""
    tangrams = sorted(list(set([entry["tangram"] for entry in this_lexicon])))

    control_trials = []
    for tangram in tangrams:
        this_option = [
            entry
            for entry in this_lexicon
            if (entry["group"] == "blue")
            and (entry["tangram"] == tangram)
            and (entry["earlier"] == "later")
        ][0]
        if this_option["shared"] == "unique":
            unseen_label_option_ = [
                entry
                for entry in this_lexicon
                if (entry["group"] == "red")
                and (entry["tangram"] == tangram)
                and (entry["earlier"] == "later")
            ][0]
        elif this_option["shared"] == "shared":
            unseen_label_option_ = [
                entry
                for entry in other_lexicon
                if (entry["tangram"] == tangram)
                and (entry["label"] != this_option["label"])
                and (entry["earlier"] == "later")
            ][0]

        # drop "shared" and "group" keys from unseen_label_option
        unseen_label_option = unseen_label_option_.copy()
        unseen_label_option.pop("shared", None)
        unseen_label_option.pop("group", None)
        unseen_label_option["group"] = "unseen"

        trial = {
            "type": "baseline",
            "goal": "refer",
            "tangram": tangram,
            "n_ingroup": 4,
            "n_outgroup": 0,
            "unseen_label": unseen_label_option["label"],
            "options": [this_option, unseen_label_option],
        }
        control_trials.append(trial)

    return control_trials


def generate_2afc_trials(item_num):
    for counterbalance in ["a", "b"]:
        with open(f"../items/item_{item_num}_{counterbalance}_lexicon.json", "r") as f:
            lexicon = json.load(f)
        other_counterbalance = "b" if counterbalance == "a" else "a"
        with open(
            f"../items/item_{item_num}_{other_counterbalance}_lexicon.json", "r"
        ) as f:
            other_lexicon = json.load(f)

        main_trials = generate_main_trials(lexicon)

        control_trials = generate_control_trials(lexicon, other_lexicon)

        trials = main_trials + control_trials
        print(
            f"Generated {len(trials)} sets of trials for item {item_num} {counterbalance}"
        )
        with open(
            f"../2AFC_trials/item_{item_num}_{counterbalance}_2AFC.json", "w"
        ) as json_file:
            json.dump(trials, json_file, indent=2)
            print(f"Saved in ../2AFC_trials/item_{item_num}_{counterbalance}_2AFC.json")


if __name__ == "__main__":
    generate_2afc_trials(item_num=0)
    generate_2afc_trials(item_num=1)
    generate_2afc_trials(item_num=2)
