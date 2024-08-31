// Selection phase
async function loadTrialInfo(item_id, participant_id) {
  const response = await fetch(`stim/2AFC_trials/item_${item_id}_2AFC.json`);
  const data = await response.json();
  const item = data[participant_id];
  if (!item) {
    throw new Error(`Item with id ${item_id} not found.`);
  }
  return item;
}

function createSelectionTrial(trial, jsPsych) {
  choice_order = jsPsych.randomization.repeat(["shared", "unique"], 1);

  function make_stimulus(trial) {
    if (trial.goal === "refer") {
      if (trial.audience === "one") {
        return `<p>Please select a description.</p>\
        <p>After the task ends, the chosen description will be shown to <b>a random player in the <span style="color: ${trial.audience_group}">${trial.audience_group}</span> group</b>.</p>\
        <p>We will ask them to use your description to <b>guess its corresponding picture</b> out of the ${params.n_tangrams} pictures, and you will be awarded a $${params.perTrialBonus.toFixed(2)} bonus if they guess the picture correctly.</p>`;
      } else if (trial.audience === "both") {
        return `<p>Please select a description.</p>\
        <p>After the task ends, the chosen description will be shown to <b>a random player across <em>both</em> groups.</b></p>\
        <p>We will ask them to use your description to <b>guess its corresponding picture</b> out of the ${params.n_tangrams} pictures, and you will be awarded a $${params.perTrialBonus.toFixed(2)} bonus if they guess the picture correctly.</p>`;
      }
    } else if (trial.goal === "social") {
      if (trial.audience === "one") {
        return `<p>Please select a description.</p>\
        <p>After the task ends, the chosen description will be shown to <b>a random player in the <span style="color: ${trial.audience_group}">${trial.audience_group}</span> group</b>.</p>\
        <p>We will ask them to use your description to <b>guess whether or not you are in their own group</b>, and you will be awarded a $${params.perTrialBonus.toFixed(2)} bonus if they correctly guess that you are in the same group as them.</p>`;
      } else if (trial.audience === "both") {
        return `<p>Please select a description.</p>\
        <p>After the task ends, the chosen description will be shown to <b>a random player across <em>both</em> groups</b>.</p>\
        <p>We will ask them to use your description to <b>guess whether or not you are in their own group</b>, and you will be awarded a $${params.perTrialBonus.toFixed(2)} bonus if they correctly guess that you are in the same group as them.</p>`;
      }
    }
  }

  return [
    {
      type: jsPsychHtmlButtonResponse,
      stimulus: make_stimulus(trial),
      choices: choice_order,
      prompt:
        "<p>Please click on the tangram-description pair when you are ready.</p>",
      button_html: (choice) =>
        `<div style="margin: 30px; cursor: pointer;" class="tangram">
                 <img src="stim/tangrams/tangram_${
                   trial[choice]?.tangram || "default"
                 }.png" style="width: 150px;" />
                 <p>${trial[choice]?.label || "default"}</p>
             </div>`,
      on_load: function () {
        document.querySelectorAll(".tangram").forEach((tangram) => {
          tangram.addEventListener("mouseover", function () {
            tangram.style.outline = "5px solid gray";
          });
          tangram.addEventListener("mouseout", function () {
            tangram.style.outline = "";
          });
        });
      },
      data: {
        trialInfo: trial,
        task: "selection",
        choice_order: choice_order,
        goal: trial.goal,
        audience: trial.audience,
        audience_group: trial.audience_group,
      },
      on_finish: function (data) {
        data.choice = data.choice_order[data.response];
      },
    },
    {
      type: jsPsychHtmlKeyboardResponse,
      stimulus: "Recording selection...",
      choices: "NO_KEYS",
      trial_duration: function () {
        return jsPsych.randomization.sampleWithoutReplacement(
          [1500, 1750, 2000, 2300],
          1
        )[0];
      },
    },
  ];
}

async function createSelectionTrials(item_id, participant_id, jsPsych) {
  try {
    const trialInfo = await loadTrialInfo(item_id, participant_id);
    const selection_phase_timeline = [];
    trialInfo.forEach((trial) => {
      const selection_trial = createSelectionTrial(trial, jsPsych);

      selection_phase_timeline.push(selection_trial);
    });
    return jsPsych.randomization.shuffle(selection_phase_timeline);
  } catch (error) {
    console.error("Error creating selection trials:", error);
    throw error;
  }
}
