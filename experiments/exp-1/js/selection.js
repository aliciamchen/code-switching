// Selection phase
async function loadTrialInfo(item_id, participant_id) {
  const response = await fetch("json/trials.json");
  const data = await response.json();
  const item = data.find((item_1) => item_1.item_id === item_id);
  if (!item) {
    throw new Error(`Item with id ${item_id} not found.`);
  }
  return item.trials[participant_id];
}

function createSelectionTrial(trial, jsPsych) {
  return [
    {
      type: jsPsychHtmlButtonResponse,
      stimulus: trial.goal === "refer" ? "Refer filler" : "Social filler",
      choices: jsPsych.randomization.repeat(["shared", "unique"], 1),
      prompt: "<p>Select a tangram-label pair to send to XXX</p>",
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
            tangram.style.outline = "5px solid lightpink";
          });
          tangram.addEventListener("mouseout", function () {
            tangram.style.outline = "";
          });
        });
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