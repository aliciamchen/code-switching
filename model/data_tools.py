"""Tools for loading and processing data
Data array, for each participant: tangram (A through L) x audience group (red vs. blue) x condition ('refer either' vs. 'refer one' vs. 'social one'). 

For each participant, only half of the tangrams have values because each participant only sees 6 tangrams. There are 6 * 2 * 3 = 36 critical trials per participant. 
"""

import numpy as np
import pandas as pd

from enums import *


def get_data():
    data = pd.read_csv("../data/3pp/earlier-later/selection_trials_clean.csv").rename(
        columns={"item_id": "tangram_set", "shared": "tangram_type"}
    )

    data.loc[data["condition"] == "either refer", "condition"] = "ReferEither"
    data.loc[data["condition"] == "one refer", "condition"] = "ReferOne"
    data.loc[data["condition"] == "one social", "condition"] = "SocialOne"

    data.loc[data["response.earlier"] == "earlier", "response.earlier"] = 1
    data.loc[data["response.earlier"] == "later", "response.earlier"] = 0

    data["response.later"] = 1 - data["response.earlier"]

    data.loc[data["tangram_type"] == "shared", "tangram_type"] = "Shared"
    data.loc[data["tangram_type"] == "unique", "tangram_type"] = "GroupSpecific"

    data.loc[data["audience_group"] == "red", "audience_group"] = "Red"
    data.loc[data["audience_group"] == "blue", "audience_group"] = "Blue"
    return data


def get_tangram_info(data):
    """For each tangram_set and counterbalance, what tangrams are available and are they group-specific or shared?"""
    tangram_info = data.groupby(
        ["tangram_set", "counterbalance", "tangram_type", "tangram"]
    ).size()
    tangram_info = tangram_info.reset_index(name="count")
    return tangram_info


def make_tangram_type_mtx(tangram_info):
    """
    Turn the output of get_tangram_info into a boolean array indicating whether the labels for each tangram are group-specific
    Output: 3 x 2 x 12 (tangram_set x counterbalance x tangram)
    """
    for _, row in tangram_info.iterrows():
        is_group_specific = np.zeros((3, 2, 12))
        tangram_set = row["tangram_set"]
        counterbalance = row["counterbalance"]
        tangram = row["tangram"]
        tangram_type = row["tangram_type"]
        is_group_specific[
            tangram_set, Counterbalance[counterbalance], Tangram[tangram]
        ] = (tangram_type == "GroupSpecific")

    return is_group_specific


def make_data_matrices(data):
    """
    For each tangram_set and counterbalance, create a matrix of responses
    Returns a dict with keys (tangram_set, counterbalance)
    and values 12 x 2 x 3 x n_participants (tangram x audience_group x condition x participant)
    """
    data_organized = {}
    for tangram_set in [0, 1, 2]:
        for counterbalance in ["a", "b"]:
            filtered_data = data[
                (data["tangram_set"] == tangram_set)
                & (data["counterbalance"] == counterbalance)
            ]
            filtered_data_participants = filtered_data["subject_id"].unique()

            this_set_mtx = np.zeros(
                (12, 2, len(Conditions), len(filtered_data_participants))
            )
            for i, participant in enumerate(filtered_data_participants):
                this_data = filtered_data[filtered_data["subject_id"] == participant]
                this_participant_mtx = np.zeros((12, 2, len(Conditions)))
                for _, row in this_data.iterrows():
                    this_participant_mtx[
                        Tangram[row["tangram"]],
                        AudienceGroup[row["audience_group"]],
                        Conditions[row["condition"]],
                    ] = row["response.later"]
                    this_set_mtx[:, :, :, i] = this_participant_mtx

            data_organized[(tangram_set, counterbalance)] = this_set_mtx

    return data_organized
