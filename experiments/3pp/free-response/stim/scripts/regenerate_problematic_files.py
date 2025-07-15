import json
import os
from manim import *
from make_videos import ChatAnimation


def regenerate_selected_tangrams():
    """Regenerate only repNum 5 videos for tangrams L, C, I, A for all items and counterbalances, using all tangrams as available_tangrams."""

    # Set up
    items_dir = os.path.join(os.path.dirname(__file__), "../items")
    item_files = [f for f in os.listdir(items_dir) if f.endswith("_game_info.json")]
    selected_tangrams = ["L", "C", "I", "A"]
    all_tangrams = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    config.media_dir = "../convo_vids"
    config.verbosity = "ERROR"

    for item_file in sorted(item_files):
        item_path = os.path.join(items_dir, item_file)
        # Parse item number and counterbalance for output naming
        basename = os.path.splitext(os.path.basename(item_file))[0]
        parts = basename.split("_")
        item_number = parts[1]
        counterbalance = parts[2]
        with open(item_path, "r") as f:
            game_info = json.load(f)
        for color, tangram_info in game_info.items():
            for tangram in selected_tangrams:
                if tangram not in tangram_info:
                    continue
                info = tangram_info[tangram]
                convo_path = f"../convos/tangram_{tangram}_game_{info['game']}.json"
                if not os.path.exists(convo_path):
                    print(f"Missing convo file: {convo_path}")
                    continue
                with open(convo_path, "r") as f:
                    convs = json.load(f)
                for conv in convs:
                    repNum = conv.get("repNum", None)
                    if repNum != 5:
                        continue
                    config.output_file = f"item_{item_number}_{counterbalance}_{color}_target_{conv['target']}_repNum_{repNum}.mp4"
                    config.quality = "low_quality"
                    print(f"Generating {config.output_file} ...")
                    scene = ChatAnimation(conv.copy(), all_tangrams.copy(), color)
                    scene.render()
                    del scene
                    import gc

                    gc.collect()
                print(
                    f"Completed {color} target {tangram} for item {item_number} {counterbalance}"
                )


if __name__ == "__main__":
    regenerate_selected_tangrams()
