import glob
import json
import csv
import uuid


def main():

    in_dir = "../../data/varied_audience/"
    filenames = sorted(glob.glob(in_dir + "raw_data" + "/*.json"))
    print(f"Found {len(filenames)} files")

    selection_trial_headers = [
        "subject_id",
        "item_id",
        "counterbalance",
        "type",
        "goal",
        "n_ingroup",
        "n_outgroup",
        "option1.tangram",
        "option1.earlier",
        "option1.length",
        "option1.label",
        "option2.tangram",
        "option2.earlier",
        "option2.length",
        "option2.label",
        "response.tangram",
        "response.earlier",
        "response.length",
        "response.group",
        "response.label",
        "understood",
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
                            "item_id": trial.get("item_id"),
                            "counterbalance": trial.get("counterbalance"),
                            "type": trial.get("trialInfo").get("type"),
                            "goal": trial.get("trialInfo").get("goal"),
                            "n_ingroup": trial.get("trialInfo").get("n_ingroup"),
                            "n_outgroup": trial.get("trialInfo").get("n_outgroup"),
                            "option1.tangram": trial.get("trialInfo")
                            .get("options")[0]
                            .get("tangram"),
                            "option1.earlier": trial.get("trialInfo")
                            .get("options")[0]
                            .get("earlier"),
                            "option1.length": trial.get("trialInfo")
                            .get("options")[0]
                            .get("length"),
                            "option1.label": trial.get("trialInfo")
                            .get("options")[0]
                            .get("label"),
                            "option2.tangram": trial.get("trialInfo")
                            .get("options")[1]
                            .get("tangram"),
                            "option2.earlier": trial.get("trialInfo")
                            .get("options")[1]
                            .get("earlier"),
                            "option2.length": trial.get("trialInfo")
                            .get("options")[1]
                            .get("length"),
                            "option2.label": trial.get("trialInfo")
                            .get("options")[1]
                            .get("label"),
                            "response.tangram": trial.get("choice").get("tangram"),
                            "response.earlier": trial.get("choice").get("earlier"),
                            "response.length": trial.get("choice").get("length"),
                            "response.group": trial.get("choice").get("group"),
                            "response.label": trial.get("choice").get("label"),
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
