// Selection phase
async function loadTrialInfo(item_id) {
  const response = await fetch(`stim/2AFC_trials/item_${item_id}_2AFC.json`);
  const data = await response.json();
  return data;
}

function createSelectionTrial(trial, jsPsych) {
  function make_stimulus(trial) {
    if (trial.goal === "refer") {
      if (trial.audience === "one") {
        return `<p>Please select a description.</p>\
        <p>Imagine that your chosen description will be shown to <strong>a random player in the <span style="color: ${trial.audience_group}">${trial.audience_group}</span> group.</strong></p>\
        <p>The question they will answer: <em>"What picture is this person referring to?"</em> <br><strong>You want them to correctly choose the corresponding picture out of the 6 pictures.</strong></p>
        <p></p>`;
      } else if (trial.audience === "both") {
        return `Not implemented`;
      }
    } else if (trial.goal === "social") {
      if (trial.audience === "one") {
        return `<p>Please select a description.</p>\
        <p>Imagine that your chosen description will be shown to <strong>a random player in the <span style="color: ${trial.audience_group}">${trial.audience_group}</span> group</strong>.</p>\
        <p>The question they will answer: <em>"Is this person sending the description also a member of the <span style="color: ${trial.audience_group}">${trial.audience_group}</span> group?"</em> <br><strong>You want them to say yes.</strong></p>
        <p></p>`;
      } else if (trial.audience === "both") {
        return `Not implemented`;
      }
    }
  }

  return [
    {
      type: jsPsychHtmlButtonResponse,
      stimulus: make_stimulus(trial),
      choices: jsPsych.randomization.repeat(trial.options, 1),
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
        choices: trial.options,
        task: "selection",
        type: "response",
        audience: trial.audience,
        audience_group: trial.audience_group,
      },
      on_finish: function (data) {
        data.choice = data.choices[data.response];
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

async function createSelectionTrials(item_id, jsPsych) {
  try {
    const trialInfo = await loadTrialInfo(item_id);
    const selection_phase_timeline = [];

    const instructions_reminder = {
      type: jsPsychHtmlButtonResponse,
      stimulus: `
      <h2>🤨 Part 2: Questions 🤨</h2>
      <h3>⚠️ Reminder ⚠️</h3>
      <div class="align-left">
      <p>
      In this part, there are a total of ${trialInfo.length} questions. For each question, you will be given two
      options to select from, each option a picture and its corresponding
      description.
      </p>
      <p>
      We will ask you to imagine that your chosen description will be sent to a certain group, \
      and that you are trying to get them to answer a question in a certain way based on the description you select. \
      We will tell you <em>which group</em> will see your description, and <em>what question</em> they will be asked to answer.
      </p>
      <p>Please make sure to read each of the questions carefully before making your selection.</p>
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
