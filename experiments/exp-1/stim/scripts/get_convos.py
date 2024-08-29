"""Extract the conversation data, across reps, for each tangram-game pair"""

import json
import numpy as np
import pandas as pd
import random
import re


def convert_int64(obj):
    """Convert int64 values to regular integers (use when saving json)"""
    if isinstance(obj, np.int64):
        return int(obj)
    return obj


def extract_letter(tangram_path: str) -> str:
    """extract letter from tangram path"""
    match = re.search(r"tangram_([A-Z])", tangram_path)
    if match:
        return match.group(1)
    return None


def get_convo_data(
    tangram: str,
    label: str,
    game: str,
    chat_data: pd.DataFrame,
    response_data: pd.DataFrame,
) -> list:
    """Get conversation data for a given tangram, label, and game

    Args:
        tangram (str): Letter representing the tangram
        label (str): Final convention label (added manually)
        game (str): ID of game
        chat_data (pd.DataFrame): chat data
        response_data (pd.DataFrame): response data

    Returns:
        list
    """
    chat = chat_data[
        (chat_data["gameId"] == game)
        & (chat_data["target"] == f"/experiment/tangram_{tangram}.png")
    ]
    response = response_data[
        (response_data["gameId"] == game)
        & (response_data["target"] == f"/experiment/tangram_{tangram}.png")
    ]

    # rename the playerIds
    # rename the speaker at rep 0 to alice, rep 1 to bob, rep 2 to carol, rep 3 to dave
    rename_mappings = {}
    for participant in chat["playerId"].unique():
        speaker_rep_num = chat[chat["speaker"] == participant]["repNum"].unique()[0]
        speaker_rep_num = (
            speaker_rep_num % 4
        )  # dont necessarily need to do this, but just in case
        rename_mappings[participant] = ["alice", "bob", "carol", "dave"][
            speaker_rep_num
        ]

    # rename the playerIds in the chat data
    for participant in rename_mappings:
        chat.loc[chat["playerId"] == participant, "playerId"] = rename_mappings[
            participant
        ]
        chat.loc[chat["speaker"] == participant, "speaker"] = rename_mappings[
            participant
        ]

    # rename the playerIds in the response data
    for participant in rename_mappings:
        response.loc[response["playerId"] == participant, "playerId"] = rename_mappings[
            participant
        ]

    # for each rep num, make convo
    output = []
    for rep in chat["repNum"].unique():
        rep_chat = chat[chat["repNum"] == rep]
        convo = []

        # extract convo data
        for idx, row in rep_chat.iterrows():
            convo.append(
                {
                    "player": row["playerId"],
                    "text": row["text"],
                    "time": random.uniform(0.5, 1.5),
                    "role": row["role"],
                }
            )

        # extract choices
        choices = response[(response["repNum"] == rep)]
        choices = {
            row["playerId"]: extract_letter(row["response"])
            for idx, row in choices.iterrows()
        }
        output.append(
            {
                "target": tangram,
                "convention": label,
                "game": game,
                "repNum": rep,
                "speakerThisRep": rep_chat["speaker"].unique()[0],
                "convo": convo,
                "choices": choices,
            }
        )

    return output


if __name__ == "__main__":
    chat_data = (
        pd.read_csv("../boyce_data/filtered_chat.csv")
        .reset_index()
        .rename(columns={"level_0": "chat_idx"})
    )
    response_data = pd.read_csv("../boyce_data/round_results.csv")

    with open("conventions_games.json", "r") as f:
        conventions_games = json.load(f)

    # for each tangram, convention, and game, extract convo data
    for tangram, conventions in conventions_games.items():
        for convention, games in conventions.items():
            for game in games:
                convo_data = get_convo_data(
                    tangram, convention, game, chat_data, response_data
                )
                with open(f"../convos/tangram_{tangram}_game_{game}.json", "w") as f:
                    json.dump(convo_data, f, default=convert_int64, indent=2)
