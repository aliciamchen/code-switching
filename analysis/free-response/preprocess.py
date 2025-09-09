import os
import json
import csv
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data/3pp/free-response/raw_data")
SELECTION_TRIALS_CSV = os.path.join(
    PROJECT_ROOT, "data/3pp/free-response/selection_trials.csv"
)
EXIT_SURVEY_CSV = os.path.join(PROJECT_ROOT, "data/3pp/free-response/exit_survey.csv")

selection_trials_fields = [
    "subject_id",
    "item_id",
    "counterbalance",
    "trial_num",
    "n_blue",
    "n_naive",
    "goal",
    "previous_selection",
    "selected_tangram",
    "selected_tangram_group",
    "selected_tangram_earlier_red",
    "selected_tangram_earlier_blue",
    "selected_tangram_later_red",
    "selected_tangram_later_blue",
    "written_label",
    "sbert_cosine_earlier_blue",
    "sbert_cosine_earlier_red",
    "sbert_cosine_later_blue",
    "sbert_cosine_later_red",
    "utt_length",
]


def extract_selection_trials(trials):
    # Load SBERT model once
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    rows = []
    for trial in tqdm(trials, desc="Processing selection trials for participant"):
        if trial.get("task") == "tangram_selection":
            info = trial.get("trialInfo", {})
            written_label = trial.get("written_label") or ""
            # Get comparison strings
            earlier_blue = trial.get("selected_tangram_earlier_blue") or ""
            earlier_red = trial.get("selected_tangram_earlier_red") or ""
            later_blue = trial.get("selected_tangram_later_blue") or ""
            later_red = trial.get("selected_tangram_later_red") or ""

            def sbert_similarity(a, b):
                if not a or not b:
                    return np.nan
                emb1 = model.encode(a, convert_to_tensor=True)
                emb2 = model.encode(b, convert_to_tensor=True)
                return float(util.cos_sim(emb1, emb2).item())

            sbert_cosine_earlier_blue = sbert_similarity(written_label, earlier_blue)
            sbert_cosine_earlier_red = sbert_similarity(written_label, earlier_red)
            sbert_cosine_later_blue = sbert_similarity(written_label, later_blue)
            sbert_cosine_later_red = sbert_similarity(written_label, later_red)
            utt_length = len(written_label.split())
            row = {
                "subject_id": trial.get("subject_id"),
                "item_id": trial.get("item_id"),
                "counterbalance": trial.get("counterbalance"),
                "trial_num": trial.get("trial_num"),
                "n_blue": info.get("n_blue"),
                "n_naive": info.get("n_naive"),
                "goal": info.get("goal"),
                "previous_selection": trial.get("previous_selection"),
                "selected_tangram": trial.get("selected_tangram"),
                "selected_tangram_group": trial.get("selected_tangram_group"),
                "selected_tangram_earlier_red": trial.get(
                    "selected_tangram_earlier_red"
                ),
                "selected_tangram_earlier_blue": trial.get(
                    "selected_tangram_earlier_blue"
                ),
                "selected_tangram_later_red": trial.get("selected_tangram_later_red"),
                "selected_tangram_later_blue": trial.get("selected_tangram_later_blue"),
                "written_label": written_label,
                "sbert_cosine_earlier_blue": sbert_cosine_earlier_blue,
                "sbert_cosine_earlier_red": sbert_cosine_earlier_red,
                "sbert_cosine_later_blue": sbert_cosine_later_blue,
                "sbert_cosine_later_red": sbert_cosine_later_red,
                "utt_length": utt_length,
            }
            rows.append(row)
    return rows


def extract_exit_survey(trials):
    # Find the last survey-multi-choice trial (exit survey)
    for trial in reversed(trials):
        if trial.get("trial_type") == "survey-html-form" and "response" in trial:
            row = {"subject_id": trial.get("subject_id")}
            for k, v in trial["response"].items():
                row[k] = v
            return row
    return None


def main():
    selection_trials = []
    exit_surveys = []
    for fname in os.listdir(RAW_DATA_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(RAW_DATA_DIR, fname), "r") as f:
                try:
                    trials = json.load(f)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
                    continue
            selection_trials.extend(extract_selection_trials(trials))
            exit_survey = extract_exit_survey(trials)
            if exit_survey:
                exit_surveys.append(exit_survey)

    # Write selection_trials.csv
    with open(SELECTION_TRIALS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selection_trials_fields)
        writer.writeheader()
        for row in selection_trials:
            writer.writerow(row)

    # Write exit_survey.csv
    if exit_surveys:
        exit_survey_fields = ["subject_id", "understood", "age", "gender", "comments"]
        with open(EXIT_SURVEY_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=exit_survey_fields)
            writer.writeheader()
            for row in exit_surveys:
                # Ensure all fields are present
                out_row = {k: row.get(k, "") for k in exit_survey_fields}
                writer.writerow(out_row)


if __name__ == "__main__":
    main()
