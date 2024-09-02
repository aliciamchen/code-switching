// exit survey and debrief
async function createExit(jsPsych) {
  try {
    // Exit survey
    const response = await fetch("html/exit-survey.html");
    const text = await response.text();
    const exit = [];
    const exit_survey = {
      type: jsPsychSurveyHtmlForm,
      preamble: `
        <p>You have reached the end of the experiment! To collect your bonus, please complete the following questions. Your answer to these questions will not affect your bonus, so please answer honestly.</p>
        `,
      html: text,
      button_label: ["Continue, save data, and collect bonus!"],
      data: { task: "exit-survey", type: "response" },
    };
    exit.push(exit_survey);

    /* Save stuff */
    function save_data_json(name, data) {
      var xhr = new XMLHttpRequest();
      xhr.open("POST", "php/save_data.php");
      xhr.setRequestHeader("Content-Type", "application/json");
      xhr.send(JSON.stringify({ filename: name, filedata: data }));
    }

    const save_data = {
      type: jsPsychCallFunction,
      func: function () {
        save_data_json(subject_id + "_output_all", jsPsych.data.get().json());
        save_data_json(
          subject_id + "_output_responses",
          jsPsych.data.get().filter({ type: "response" }).json()
        );
      },
      timing_post_trial: 0,
    };

    if (!local_testing) {
      exit.push(save_data);
    }

    // Debrief
    const debrief = {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: `<p>Thanks for participating in the experiment!</p>
                    <p><a href="https://app.prolific.co/submissions/complete?cc=C1E1PWV8">Click here to return to Prolific and complete the study</a>.</p>
                    <p>It is now safe to close the window. Your pay will be delivered within a few days.</p>
                    `,
      choices: "NO_KEYS",
    };
    exit.push(debrief);

    return exit;
  } catch (error) {
    console.error("Error creating exit survey:", error);
    throw error;
  }
}
