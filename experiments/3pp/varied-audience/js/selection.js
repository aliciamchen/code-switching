// Selection phase
async function loadTrialInfo(item_id, counterbalance) {
  const response = await fetch(`stim/2AFC_trials/item_${item_id}_${counterbalance}_2AFC.json`);
  const data = await response.json();
  return data;
}


async function createSelectionTrials(item_id, counterbalance, jsPsych) {
  try {
    const trialInfo = await loadTrialInfo(item_id, counterbalance);
    const selection_phase_timeline = [];

    const instructions_reminder = {
      type: jsPsychHtmlButtonResponse,
      stimulus: `
      <h2>🤨 Part 2: Questions 🤨</h2>
      <h3>⚠️ Reminder ⚠️</h3>
      <div class="align-left">
      <p>
        In the second part of the task, it will be <strong>up to you</strong> to
        choose words that go with these pictures.
      </p>
      <p>
        On each screen, you will see two labels describing a picture, and
        you will have to pick one of them. We will tell you
        <strong>what audience</strong> will see your choice (a number of blue-group members, 
        and participants who played this game in a separate, unspecified group) 
        and <strong>what question</strong> they will be answering when
        they see it. Your goal is to try to get them to answer the question in a
        certain way based on the label you select.
      </p>
      <p>
        Please read carefully, so you know <strong>who</strong> will see the picture
        and words you chose, <strong>what question</strong> they will be answering,
        and <strong>how</strong> you want them to answer it.
      </p>
      <p>You will do this 66 times. You will receive a bonus of $0.10 for each correct choice.</p>
      </div>
      `,
      choices: ["Continue"],
    };

    selection_phase_timeline.push(instructions_reminder);

    selection_trials = [];
    trialInfo.forEach((trial) => {
      const selection_trial = createSelectionTrial(trial, jsPsych);

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

function createSelectionTrial(trial, jsPsych) {
  trial.nIngroup = 4;
  trial.nOutgroup = 0;
  // TODO: make sure this info is in trial param

  var shuffled_options = jsPsych.randomization.repeat(trial.options, 1);
  return [
    {
      type: jsPsychHtmlButtonResponse,
      stimulus: generateStimulus(trial),
      choices: shuffled_options,
      prompt:
        "<p>Please click on the picture-description pair when you are ready.</p>",
      button_html: (choice) =>
        `<div style="margin: 30px; cursor: pointer;" class="tangram">
                 <img src="stim/tangrams/tangram_${
                   choice?.tangram || "default"
                 }.png" style="width: 150px;" />
                 <p>${choice?.label || "default"}</p>
             </div>`,
      data: {
        trialInfo: trial,
        displayed_options: shuffled_options,
        task: "selection",
      },
      on_finish: function (data) {
        data.choice = data.displayed_options[data.response];
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
