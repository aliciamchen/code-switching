import glob
import json
import csv
import uuid


def main():

    in_dir = "../../data/pilot-2_2024-09-18"
    filenames = sorted(glob.glob(in_dir + "/*.json"))

    selection_trial_headers = [
        "subject_id",
        "item_id",
        "type",
        "goal",
        "audience",
        "audience_group",
        "option1.tangram",
        "option1.shared",
        "option1.group",
        "option1.label",
        "option2.tangram",
        "option2.shared",
        "option2.group",
        "option2.label",
        "response.tangram",
        "response.shared",
        "response.group",
        "response.label",
        "understood",
    ]

    exit_survey_headers = [
        "subject_id",
        "understood",
        "age",
        "gender",
        "feedback",
        "comments",
    ]

    subject_id_map = {}

    def get_anonymized_id(original_id):
        if original_id not in subject_id_map:
            subject_id_map[original_id] = str(uuid.uuid4())
        return subject_id_map[original_id]

    with open("selection_trials.csv", "w", newline="") as csvfile:
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
                            "type": trial.get("trialInfo").get("type"),
                            "goal": trial.get("trialInfo").get("goal"),
                            "audience": trial.get("trialInfo").get("audience"),
                            "audience_group": trial.get("trialInfo").get("audience_group"),
                            "option1.tangram": trial.get("trialInfo")
                            .get("options")[0]
                            .get("tangram"),
                            "option1.shared": trial.get("trialInfo")
                            .get("options")[0]
                            .get("shared"),
                            "option1.group": trial.get("trialInfo")
                            .get("options")[0]
                            .get("group"),
                            "option1.label": trial.get("trialInfo")
                            .get("options")[0]
                            .get("label"),
                            "option2.tangram": trial.get("trialInfo")
                            .get("options")[1]
                            .get("tangram"),
                            "option2.shared": trial.get("trialInfo")
                            .get("options")[1]
                            .get("shared"),
                            "option2.group": trial.get("trialInfo")
                            .get("options")[1]
                            .get("group"),
                            "option2.label": trial.get("trialInfo")
                            .get("options")[1]
                            .get("label"),
                            "response.tangram": trial.get("choice").get("tangram"),
                            "response.shared": trial.get("choice").get("shared"),
                            "response.group": trial.get("choice").get("group"),
                            "response.label": trial.get("choice").get("label"),
                            "understood": exit_survey.get("response").get("understood"),
                        }
                    )

    with open("exit_survey.csv", "w", newline="") as csvfile:
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
                        "feedback": exit_survey.get("response").get("feedback"),
                        "comments": exit_survey.get("response").get("comments"),
                    }
                )


if __name__ == "__main__":
    main()
