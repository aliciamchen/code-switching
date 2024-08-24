async function loadInstructionsFromHTML() {
  try {
    const response = await fetch("html/instructions.html");
    const text = await response.text();
    return text.split("<!-- PAGE BREAK -->");
  } catch (error) {
    console.error("Error loading instructions:", error);
    throw error;
  }
}

async function createInstructions() {
  try {
    const instructionPages = await loadInstructionsFromHTML();
    return {
      type: jsPsychInstructions,
      pages: instructionPages,
      show_clickable_nav: true,
    };
  } catch (error) {
    console.error("Error creating instructions:", error);
    throw error;
  }
}

// Create a comprehension check quiz
async function createComprehensionCheck() {
  try {
    const response = await fetch("json/comprehension.json");
    const text = await response.text();
    const questions = JSON.parse(text);
    return {
      type: jsPsychSurveyMultiChoice,
      questions: questions,
      preamble: "<p>Please answer the following questions:</p>",
      on_finish: function (data) {
        data.pass = [
          data.response.Q0.includes("stem"),
          data.response.Q1.includes("accurately"),
          data.response.Q2.includes("new"),
        ].every(Boolean);

        if (!data.pass) {
          failCount++;
          if (failCount >= 3) {
            jsPsych.abortExperiment(
              "You have failed the comprehension check three times. You cannot proceed with the experiment."
            );
          }
        }
      },
    };
  } catch (error) {
    console.error("Error creating comprehension check:", error);
    throw error;
  }
}

function createFailComprehensionCheck() {
  return {
    timeline: [
      {
        type: jsPsychHtmlButtonResponse,
        stimulus: function () {
          return `<p>Oops, you have missed question(s) on the comprehension check! We'll show you the instructions again.</p>
        <p>${3 - failCount} ${failCount != 2 ? `tries` : `try`} left</p>`;
        },
        choices: ["Got it!"],
      },
    ],
    conditional_function: function () {
      const responses = jsPsych.data.getLastTrialData();
      return !responses.select("pass").values[0];
    },
  };
}

async function createComprehensionLoop(jsPsych) {
  const instructions = await createInstructions();
  const comprehensionCheck = await createComprehensionCheck();
  const failComprehensionCheck = createFailComprehensionCheck();

  return {
    timeline: [instructions, comprehensionCheck, failComprehensionCheck],
    loop_function: function (data) {
      const passed = data.select("pass").values[0];
      return !passed;
    },
  };
}
