"""
For each time a red-specific tangram is selected for the social goal:
- find the item_id, counterbalance pair in `converged_expressions.csv` that matches the item_id, counterbalance pair in the selection trial
- calculate similarity of the written description to each of the blue-group converged expressions for all of the blue-group tangrams (blue-specific + shared)
- calculate similarity of the written description to each of the red-group converged expressions for all of the red-group tangrams (red-specific + shared)
- select the most similar later description among each of the blue-group tangrams
- select the most similar later description among each of the red-group tangrams
- average the similarity scores to each of the blue-group tangrams
- average the similarity scores to each of the red-group tangrams

And also calculate the same for blue-specific tangrams, for comparison
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
converged_expressions_path = os.path.join(
    PROJECT_ROOT, "data/3pp/free-response/converged_expressions.csv"
)
print(f"Loading converged expressions from {converged_expressions_path}")
# Load converged expressions
converged_expressions = pd.read_csv(converged_expressions_path)

# Prepare SBERT model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


def sbert_similarity(a, b):
    if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
        return np.nan
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())


# Function to filter relevant rows from large CSV in chunks
def get_relevant_trials(csv_path):
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=10000):
        filtered = chunk[
            (chunk["goal"] == "social")
            & (
                (chunk["selected_tangram_group"] == "red_specific")
                | (chunk["selected_tangram_group"] == "blue_specific")
            )
        ]
        if not filtered.empty:
            chunks.append(filtered)
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    else:
        return pd.DataFrame()


# Get relevant trials
selection_trials_path = os.path.join(
    PROJECT_ROOT, "data/3pp/free-response/selection_trials.csv"
)
social_trials = get_relevant_trials(selection_trials_path)


results = []
print(f"Number of tangram selections goal: {len(social_trials)}")

for idx, trial in tqdm(
    social_trials.iterrows(), total=len(social_trials), desc="Processing trials"
):
    item_id = trial["item_id"]
    counterbalance = trial["counterbalance"]
    written_label = trial["written_label"]
    # Get converged expressions for this item_id/counterbalance
    exprs = converged_expressions[
        (converged_expressions["item_id"] == item_id)
        & (converged_expressions["counterbalance"] == counterbalance)
    ]
    # Blue-group tangrams: blue_specific + shared
    blue_exprs = exprs[
        exprs["selected_tangram_group"].isin(["blue_specific", "shared"])
    ]
    # Red-group tangrams: red_specific + shared
    red_exprs = exprs[exprs["selected_tangram_group"].isin(["red_specific", "shared"])]
    # For each blue-group tangram, get the later blue description
    blue_sims = []
    for _, row in blue_exprs.iterrows():
        label = row["selected_tangram_later_blue"]
        if pd.notnull(label):
            sim = sbert_similarity(written_label, label)
            blue_sims.append(sim)
    # For each red-group tangram, get the later red description
    red_sims = []
    for _, row in red_exprs.iterrows():
        label = row["selected_tangram_later_red"]
        if pd.notnull(label):
            sim = sbert_similarity(written_label, label)
            red_sims.append(sim)
    # Most similar among blue and red
    max_blue_sim = max(blue_sims) if blue_sims else np.nan
    max_red_sim = max(red_sims) if red_sims else np.nan
    # Average similarity
    mean_blue_sim = np.nanmean(blue_sims) if blue_sims else np.nan
    mean_red_sim = np.nanmean(red_sims) if red_sims else np.nan
    results.append(
        {
            "subject_id": trial["subject_id"],
            "item_id": item_id,
            "counterbalance": counterbalance,
            "trial_num": trial["trial_num"],
            "selected_tangram_group": trial["selected_tangram_group"],
            "written_label": written_label,
            "max_blue_sim": max_blue_sim,
            "max_red_sim": max_red_sim,
            "mean_blue_sim": mean_blue_sim,
            "mean_red_sim": mean_red_sim,
        }
    )

results_df = pd.DataFrame(results)

# Save detailed results
results_df.to_csv(
    os.path.join(
        PROJECT_ROOT, "analysis/3pp/free-response/red_social_similarity_results.csv"
    ),
    index=False,
)
