import json
import os
from manim import *
from make_videos import ChatAnimation

def regenerate_problematic_files():
    """Regenerate only repNum 5 for the problematic targets"""
    
    with open("../items/item_0_a_game_info.json", "r") as f:
        game_info = json.load(f)
    
    config.media_dir = "../convo_vids"
    config.verbosity = "ERROR"
    
    # Define the problematic targets based on current issue
    problematic_targets = {
        "blue": "C",  # 3rd blue target
        "red": "I"    # Current problematic red target
    }
    
    for color, target in problematic_targets.items():
        print(f"Regenerating {color} target {target} repNum 5...")
        
        # Get target info
        target_info = game_info[color][target]
        
        with open(f"../convos/tangram_{target}_game_{target_info['game']}.json", "r") as f:
            convs = json.load(f)
        
        available_tangrams = list(game_info[color].keys())
        
        for conv in convs:
            if conv['repNum'] == 5:
                config.output_file = f"item_0_a_{color}_target_{conv['target']}_repNum_{conv['repNum']}.mp4"
                config.quality = "low_quality"
                
                print(f"  Generating repNum {conv['repNum']}...")
                
                # Create fresh scene
                scene = ChatAnimation(conv, available_tangrams, color)
                scene.render()
                
                # Clean up
                del scene
                import gc
                gc.collect()
                break  # Only process repNum 5
        
        print(f"Completed {color} target {target} repNum 5")

if __name__ == "__main__":
    regenerate_problematic_files() 