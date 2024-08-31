// exit survey and debrief
async function createExit() {
  try {
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
    };
    exit.push(exit_survey);

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
