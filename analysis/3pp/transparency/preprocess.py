import glob
import json
import csv
import uuid


def main():

    in_dir = "../../../data/3pp/transparency/"
    filenames = sorted(glob.glob(in_dir + "raw_data" + "/*.json"))

    selection_trial_headers = [
        "subject_id",
        "tangram_set",
        "target", 
        "earlier", 
        "length", 
        "label",
        "response",
        "correct",
        "understood"
    ]

    exit_survey_headers = [
        "subject_id",
        "understood",
        "age",
        "gender",
        "comments",
    ]

    subject_id_map = {}

    def get_anonymized_id(original_id):
        if original_id not in subject_id_map:
            subject_id_map[original_id] = str(uuid.uuid4())
        return subject_id_map[original_id]

    with open(in_dir + "selection_trials.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=selection_trial_headers)
        writer.writeheader()

        for filename in filenames:

            with open(filename, "r") as f:
                raw_json_data = json.load(f)

                selection_trials = [
                    trial for trial in raw_json_data if trial.get("task") == "selection"
                ]
                exit_survey = [
                    trial
                    for trial in raw_json_data
                    if trial.get("task") == "exit-survey"
                ][0]

                for trial in selection_trials:

                    writer.writerow(
                        {
                            "subject_id": get_anonymized_id(trial.get("subject_id")),
                            "tangram_set": trial.get("trialInfo").get("tangram_set"),
                            "target": trial.get("trialInfo").get("tangram"),
                            "earlier": trial.get("trialInfo").get("earlier"),
                            "length": trial.get("trialInfo").get("length"),
                            "label": trial.get("trialInfo").get("label"),
                            "response": trial.get("choice"),
                            "correct": trial.get("correct"),
                            "understood": exit_survey.get("response").get("understood"),
                        }
                    )

    print(f"Selection trials saved in {in_dir}selection_trials.csv")

    with open(in_dir + "exit_survey.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=exit_survey_headers)
        writer.writeheader()

        for filename in filenames:

            with open(filename, "r") as f:
                raw_json_data = json.load(f)

                exit_survey = [
                    trial
                    for trial in raw_json_data
                    if trial.get("task") == "exit-survey"
                ][0]

                writer.writerow(
                    {
                        "subject_id": get_anonymized_id(exit_survey.get("subject_id")),
                        "understood": exit_survey.get("response").get("understood"),
                        "age": exit_survey.get("response").get("age"),
                        "gender": exit_survey.get("response").get("gender"),
                        "comments": exit_survey.get("response").get("comments"),
                    }
                )

    print(f"Exit survey responses saved in {in_dir}exit_survey.csv")

if __name__ == "__main__":
    main()
