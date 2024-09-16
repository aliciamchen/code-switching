import itertools
import random
import json

random.seed(88)


def separate_tangrams(lexicon):
    shared_tangrams = [entry for entry in lexicon if entry["shared"] == "shared"]
    unique_tangrams = [entry for entry in lexicon if entry["shared"] == "unique"]
    return shared_tangrams, unique_tangrams


def generate_main_trials(lexicon, shared_tangrams, unique_tangrams):
    unique_set = sorted(list(set([entry["tangram"] for entry in unique_tangrams])))
    shared_set = sorted(list(set([entry["tangram"] for entry in shared_tangrams])))

    # generate list of all possible pairs of tangram, where there is one shared and one unique tangram
    tangram_pairs = list(itertools.product(shared_set, unique_set))

    trials = []
    for tangram_pair in tangram_pairs:
        for audience_group in ["red", "blue"]:
            for goal in ["refer", "social"]:
                trial = {
                    "type": "diff",  # the two tangrams are different
                    "goal": goal,
                    "audience": "one",
                    "audience_group": audience_group,
                    "options": [
                        [
                            entry
                            for entry in lexicon
                            if (entry["group"] == audience_group)
                            and (entry["tangram"] == tangram_pair[0])
                        ][0],
                        [
                            entry
                            for entry in lexicon
                            if (entry["group"] == audience_group)
                            and (entry["tangram"] == tangram_pair[1])
                        ][0],
                    ],
                }
                trials.append(trial)
    assert len(trials) == 36
    return trials


def generate_control_trials(
    this_lexicon, other_lexicon, shared_tangrams, unique_tangrams
):
    """other_lexicon is lexicon of another (counterbalanced) items to get the labels that are not seen"""
    unique_set = list(set([entry["tangram"] for entry in unique_tangrams]))
    shared_set = list(set([entry["tangram"] for entry in shared_tangrams]))

    shared_control_trials = []
    for shared_tangram in shared_set:
        for audience_group in ["red", "blue"]:
            this_option = [
                entry
                for entry in this_lexicon
                if (entry["group"] == audience_group)
                and (entry["tangram"] == shared_tangram)
            ][0]
            unseen_label_option_ = [
                entry
                for entry in other_lexicon
                if (entry["tangram"] == shared_tangram)
                and (entry["label"] != this_option["label"])
            ][0]

            # drop "shared" and "group" keys from unseen_label_option
            unseen_label_option = unseen_label_option_.copy()
            unseen_label_option.pop("shared", None)
            unseen_label_option.pop("group", None)
            unseen_label_option["group"] = "unseen"

            trial = {
                "type": "same",
                "goal": "refer",
                "audience": "one",
                "unseen_label": unseen_label_option["label"],
                "audience_group": audience_group,
                "options": [this_option, unseen_label_option],
            }
            shared_control_trials.append(trial)

    unique_control_trials = []
    for unique_tangram in unique_set:
        for audience_group in ["red", "blue"]:
            red_label_option = [
                entry
                for entry in this_lexicon
                if (entry["group"] == "red") and (entry["tangram"] == unique_tangram)
            ][0]
            blue_label_option = [
                entry
                for entry in this_lexicon
                if (entry["group"] == "blue") and (entry["tangram"] == unique_tangram)
            ][0]

            trial = {
                "type": "same",
                "goal": "social",
                "audience": "one",
                "audience_group": audience_group,
                "options": [red_label_option, blue_label_option],
            }
            unique_control_trials.append(trial)

    return shared_control_trials + unique_control_trials


def main(items):
    for item_num in items:
        with open(f"../items/item_{item_num}_lexicon.json", "r") as f:
            lexicon = json.load(f)
        other_item_num = 1 if item_num == 0 else 0
        with open(f"../items/item_{other_item_num}_lexicon.json", "r") as f:
            other_lexicon = json.load(f)

        shared_tangrams, unique_tangrams = separate_tangrams(lexicon)
        main_trials = generate_main_trials(lexicon, shared_tangrams, unique_tangrams)
        control_trials = generate_control_trials(
            lexicon, other_lexicon, shared_tangrams, unique_tangrams
        )
        trials = main_trials + control_trials
        print(f"Generated {len(trials)} sets of trials for item {item_num}")
        with open(f"../2AFC_trials/item_{item_num}_2AFC.json", "w") as json_file:
            json.dump(trials, json_file, indent=2)
            print(f"Saved in ../2AFC_trials/item_{item_num}_2AFC.json")


if __name__ == "__main__":
    main([0, 1])
    print("Done")
