// Selection phase
async function loadUtterances() {
  const response = await fetch(`all_utterances.json`);
  const data = await response.json();
  return data;
}

async function loadTrials() {
  const response = await fetch(`trials.json`);
  const data = await response.json();
  return data;
}

function createSelectionTrial(utterance, jsPsych) {
  var shuffled_options = jsPsych.randomization.repeat(utterance.context, 1);

  return [
    {
      type: jsPsychHtmlButtonResponse,
      stimulus: `<em>${utterance.label}</em><p></p>`,
      choices: shuffled_options,
      button_layout: "grid",
      grid_rows: 2, 
      grid_columns: 3, 
      prompt:
        "<p>Please read the description carefully and then select the picture that best matches the description.</p>",
      button_html: (choice) =>
        `<div class="tangram">
                 <img src="tangrams/tangram_${
                   choice || "default"
                 }.png" style="width: 150px;" />
             </div>`,
      data: {
        trialInfo: utterance,
        displayed_options: shuffled_options,
        task: "selection",
      },
      on_finish: function (data) {
        data.choice = data.displayed_options[data.response];
        data.correct = data.choice === utterance.tangram;
      },
    },
    {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: "Recording selection...",
      choices: "NO_KEYS",
      trial_duration: function () {
        return jsPsych.randomization.sampleWithoutReplacement(
          [1500, 1750],
          1
        )[0];
      },
    },
  ];
}

async function createSelectionTrials(jsPsych) {
  try {
    const trialInfo = await loadTrials();
    const all_utterances = await loadUtterances();

    const selection_phase_timeline = [];

    selection_trials = [];
    trialInfo.forEach((trial) => {
      utterances = all_utterances.filter(
        (utterance) =>
          utterance.tangram_set === trial.tangram_set &&
          utterance.tangram === trial.tangram
      );
      // sample one utterance
      utterance = jsPsych.randomization.sampleWithoutReplacement(
        utterances,
        1
      )[0];
      const selection_trial = createSelectionTrial(utterance, jsPsych);

      selection_trials.push(selection_trial);
    });
    selection_trials_randomized =
      jsPsych.randomization.shuffle(selection_trials);

    selection_phase_timeline.push(selection_trials_randomized);

    return selection_phase_timeline;
  } catch (error) {
    console.error("Error creating selection trials:", error);
    throw error;
  }
}
