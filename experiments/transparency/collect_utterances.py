# Collect all the earlier-later utterances from the 3pp experiment into a single file
import json

tangram_sets = [0, 1, 2]  # earlier I named this "item"
counterbalances = ["a", "b"]
contexts = {
    0: ["A", "B", "C", "D", "H", "L"],
    1: ["E", "F", "G", "I", "J", "K"],
    2: ["A", "C", "E", "G", "I", "K"]
}  # indexed by tangram_set

trials = []
for tangram_set, context in contexts.items():
    for tangram in context:
        trials.append({
            "tangram_set": tangram_set,
            "tangram": tangram
        })

with open("trials.json", "w") as f:
    json.dump(trials, f, indent=2)

all_utterances = []

for tangram_set in tangram_sets:
    for counterbalance in counterbalances:
        with open(
            f"../earlier-later/stim/items/item_{tangram_set}_{counterbalance}_lexicon.json",
            "r",
        ) as f:
            lexicon = json.load(f)
            for utterance in lexicon:
                utterance["tangram_set"] = tangram_set
                del utterance["group"]
                del utterance["game"]
                del utterance["shared"]

            all_utterances.extend(lexicon)

# sort all_utterances by tangram_set, tangram, and earlier
all_utterances.sort(key=lambda x: (x["tangram_set"], x["tangram"], x["earlier"]))

# remove duplicate entries
seen = set()
unique_utterances = []
for utterance in all_utterances:
    dict_tuple = tuple(utterance.items())
    if dict_tuple not in seen:
        seen.add(dict_tuple)
        unique_utterances.append(utterance)

# add context info
for utterance in unique_utterances:
    utterance["context"] = contexts[utterance["tangram_set"]]

with open("all_utterances.json", "w") as f:
    json.dump(unique_utterances, f, indent=2)
