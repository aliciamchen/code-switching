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
        <p>After the task ends, the chosen description will be shown to <b>a random player in the <span style="color: ${
          trial.audience_group
        }">${trial.audience_group}</span> group</b>.</p>\
        <p>We will ask them to use your description to <b>guess its corresponding picture</b> out of the ${
          params.n_tangrams
        } pictures, and you will be awarded a $${params.perTrialBonus.toFixed(
          2
        )} bonus if they guess the picture correctly.</p>`;
      } else if (trial.audience === "both") {
        return `<p>Please select a description.</p>\
        <p>After the task ends, the chosen description will be shown to <b>a random player across <em>both</em> groups.</b></p>\
        <p>We will ask them to use your description to <b>guess its corresponding picture</b> out of the ${
          params.n_tangrams
        } pictures, and you will be awarded a $${params.perTrialBonus.toFixed(
          2
        )} bonus if they guess the picture correctly.</p>`;
      }
    } else if (trial.goal === "social") {
      if (trial.audience === "one") {
        return `<p>Please select a description.</p>\
        <p>After the task ends, the chosen description will be shown to <b>a random player in the <span style="color: ${
          trial.audience_group
        }">${trial.audience_group}</span> group</b>.</p>\
        <p>We will ask them to use your description to <b>guess whether or not you are in their own group</b>, and you will be awarded a $${params.perTrialBonus.toFixed(
          2
        )} bonus if they correctly guess that you are in the same group as them.</p>`;
      } else if (trial.audience === "both") {
        return `<p>Please select a description.</p>\
        <p>After the task ends, the chosen description will be shown to <b>a random player across <em>both</em> groups</b>.</p>\
        <p>We will ask them to use your description to <b>guess whether or not you are in their own group</b>, and you will be awarded a $${params.perTrialBonus.toFixed(
          2
        )} bonus if they correctly guess that you are in the same group as them.</p>`;
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
        type: "response",
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

    const instructions_reminder = {
      type: jsPsychHtmlButtonResponse,
      stimulus: `
      <h2>🤨 Part 2: Questions 🤨</h2>
      <h3>⚠️ Reminder ⚠️</h3>
      <div class="align-left">
      <p>
      In this part, there are a total of 12 questions. For each question, you will be given two
      options to select from, each option a picture and its corresponding
      description.
      </p>
      <p>
        The descriptions you choose will be sent to the participants that played the
        game you just observed. For each pair of options, we will tell you (1)
        <em>who</em> will see the desciption you select; and (2) <em>what</em> we
        will prompt the players to guess based on the description you select.
      </p>
      <p>
        You will earn an
        additional bonus, up to XXX, based on the number of correct answers the
        players select using your responses in this part of the task.
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
