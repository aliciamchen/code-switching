import itertools
import random
import json

random.seed(88)


def separate_tangrams(lexicon):
    shared_tangrams = [entry for entry in lexicon if entry["shared"] == "shared"]
    unique_tangrams = [entry for entry in lexicon if entry["shared"] == "unique"]
    return shared_tangrams, unique_tangrams


def generate_all_unique_sets(lexicon, shared_tangrams, unique_tangrams):
    """Generate all unique sets of trials"""

    # what are the tangrams in 'unique'?
    unique_set = list(set([entry["tangram"] for entry in unique_tangrams]))
    shared_set = list(set([entry["tangram"] for entry in shared_tangrams]))

    all_sets = []
    shared_permutations = list(itertools.permutations(shared_set))

    # Generate all combinations of the valid pairs of audiences
    audience_pairs = [("one", "both"), ("both", "one")]
    audience_combs = list(itertools.product(audience_pairs, repeat=3))
    # get rid of the combinations that have three of the same element
    audience_combs = [
        audience_comb
        for audience_comb in audience_combs
        if len(set(audience_comb)) == 2
    ]

    for shared_perm in shared_permutations:
        for comb in audience_combs:
            # print(comb)
            trials = []
            for i in range(3):
                trial_red_refer = {
                    "goal": "refer",
                    "audience": comb[i][0],
                    "audience_group": "red",
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
                trial_red_social = {
                    "goal": "social",
                    "audience": comb[i][0],
                    "audience_group": "red",
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
                trial_blue_refer = {
                    "goal": "refer",
                    "audience": comb[i][1],
                    "audience_group": "blue",
                    "shared": [
                        entry
                        for entry in lexicon
                        if (entry["group"] == "blue")
                        and (entry["tangram"] == shared_perm[i])
                    ][0],
                    "unique": [
                        entry
                        for entry in lexicon
                        if (entry["group"] == "blue")
                        and (entry["tangram"] == unique_set[i])
                    ][0],
                }
                trial_blue_social = {
                    "goal": "social",
                    "audience": comb[i][1],
                    "audience_group": "blue",
                    "shared": [
                        entry
                        for entry in lexicon
                        if (entry["group"] == "blue")
                        and (entry["tangram"] == shared_perm[i])
                    ][0],
                    "unique": [
                        entry
                        for entry in lexicon
                        if (entry["group"] == "blue")
                        and (entry["tangram"] == unique_set[i])
                    ][0],
                }
                trials.append(trial_red_refer)
                trials.append(trial_red_social)
                trials.append(trial_blue_refer)
                trials.append(trial_blue_social)
            all_sets.append(trials)
    return all_sets


def main(n_items):
    for item_num in range(n_items):
        with open(f"../items/item_{item_num}_lexicon.json", "r") as f:
            lexicon = json.load(f)

        shared_tangrams, unique_tangrams = separate_tangrams(lexicon)
        all_sets = generate_all_unique_sets(lexicon, shared_tangrams, unique_tangrams)
        print(f"Generated {len(all_sets)} sets of trials for item {item_num}")
        with open(f"../2AFC_trials/item_{item_num}_2AFC.json", "w") as json_file:
            # note that each index of this list is a set of trials for a single participant
            json.dump(all_sets, json_file, indent=2)
            print(f"Saved in ../2AFC_trials/item_{item_num}_2AFC.json")


if __name__ == "__main__":
    main(10)
    print("Done")
