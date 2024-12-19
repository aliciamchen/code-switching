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
        <p>You have reached the end of the experiment! Thank you for your participation.</p>
        <p><strong>To collect your pay, please complete the following questions. Your answer to these questions will not affect your pay, so please answer honestly.</strong></p>
        `,
      html: text,
      button_label: ["Continue, save data, and collect pay!"],
      data: { task: "exit-survey", type: "response" }
    };
    exit.push(exit_survey);

    const save_data = {
      type: jsPsychPipe,
      action: "save",
      experiment_id: "x15GZNL2nIef",
      filename: `${subject_id}.json`,
      data_string: () => jsPsych.data.get().json(),
    };

    if (!local_testing) {
      exit.push(save_data);
    }

    // Debrief
    const debrief = {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: function () {
        var totalCorrect = jsPsych.data.get().select("correct").sum();
        var totalPay = 2 + totalCorrect * 0.1;
        return `<p>Thanks for participating in the experiment!</p>
        <p>You got <b>${totalCorrect}</b> out of 18 responses correct.</p>
        <p>Your total pay is <b>${totalPay.toFixed(2)}</b>.</p>
        <p><a href="https://app.prolific.com/submissions/complete?cc=C1AKB1HA">Click here to return to Prolific and complete the study</a>.</p>
        <p>It is now safe to close the window. Your pay will be delivered within a few days.</p>
        `;
      },
      choices: "NO_KEYS",
    };
    exit.push(debrief);

    return exit;
  } catch (error) {
    console.error("Error creating exit survey:", error);
    throw error;
  }
}
