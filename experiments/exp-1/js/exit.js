// exit survey
async function createExitSurvey() {
  try {
    const response = await fetch("html/exit-survey.html");
    const text = await response.text();
    return {
      type: jsPsychSurveyHtmlForm,
      preamble: `
        <p>You have reached the end of the experiment! To collect your bonus, please complete the following questions. Your answer to these questions will not affect your bonus, so please answer honestly.</p>
        `,
      html: text,
      button_label: ["Continue, save data, and collect bonus!"],
    };
  } catch (error) {
    console.error("Error creating exit survey:", error);
    throw error;
  }
}
