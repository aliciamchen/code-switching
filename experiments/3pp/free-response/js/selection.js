// Free-response selection phase

// Create trial conditions for the free-response experiment
async function createFreeResponseTrialInfo(item_id, counterbalance, jsPsych) {
  const response = await fetch(
    `stim/items/item_${item_id}_${counterbalance}_game_info.json`
  );
  const data = await response.json();

  // Get all available tangrams
  const blue_tangrams = Object.keys(data.blue);
  const red_tangrams = Object.keys(data.red);
  const all_tangrams = [...new Set([...blue_tangrams, ...red_tangrams])];

  // Categorize tangrams by group
  const blue_specific_tangrams = blue_tangrams.filter(
    (t) => data.blue[t].group === "blue_specific"
  ); // Blue only
  const shared_tangrams = blue_tangrams.filter(
    (t) => data.blue[t].group === "shared"
  ); // Shared between red and blue
  const red_specific_tangrams = red_tangrams.filter((t) => data.red[t].group === "red_specific"); // Red only

  // Create audience conditions matching make_2afc_trials.py
  const ns_ingroup = [0, 1, 2, 3, 4];
  const ns_outgroup = [0, 1, 2, 4, 8, 16];

  const audienceConditions = [];

  // Generate all combinations except (0, 0)
  for (const n_blue of ns_ingroup) {
    for (const n_naive of ns_outgroup) {
      if (!(n_blue === 0 && n_naive === 0)) {
        // Add referential goal condition
        audienceConditions.push({ n_blue, n_naive, goal: "refer" });

        // Add social goal condition only when blue group members are present
        if (n_blue > 0) {
          audienceConditions.push({ n_blue, n_naive, goal: "social" });
        }
      }
    }
  }

  // Create trials by assigning tangrams to conditions
  const trials = [];

  audienceConditions.forEach((condition, conditionIndex) => {
    const availableTangrams = [...all_tangrams];
    const trial = {
      n_blue: condition.n_blue,
      n_naive: condition.n_naive,
      goal: condition.goal,
      available_tangrams: availableTangrams,
      blue_specific_tangrams: blue_specific_tangrams,
      shared_tangrams: shared_tangrams,
      red_specific_tangrams: red_specific_tangrams,
      convo_info: data,
    };
    trials.push(trial);
  });

  return trials;
}

// Global variable to track previous selection
let previous_selection = null;

// Helper: Generate a single selection trial
function makeFreeResponseSelectionTrial(trial, trial_index) {
  // Exclude previously selected tangram if needed
  let availableTangrams = trial.available_tangrams;
  if (previous_selection) {
    availableTangrams = availableTangrams.filter(
      (t) => t !== previous_selection
    );
  }

  return {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function () {
      let html = `
        <div><span style="font-weight:bold; font-size:1.1em;">Your Audience:</span></div>
        <div>${generateAudienceSVG(
          trial.n_blue,
          trial.n_naive
        )}${generateFreeResponseStimulusCount(
        trial.n_blue,
        trial.n_naive
      )}</div>
        <div style="margin: 35px 0 10px 0; background: #f3eafd; padding: 16px 18px; border-radius: 10px; font-size: 1.08em; font-weight: 500; box-shadow: 0 2px 8px rgba(142,68,173,0.06);">${generateFreeResponseStimulusText(
          trial.goal
        )}</div>
        <label for="label-input" style="font-size: 1em; font-weight: bold; margin: 20px 0 0 0; display: block;"><strong>Choose a picture:</strong></label>
        <div id="tangram-grid" style="display: grid; grid-template-columns: repeat(4, 1fr); grid-template-rows: repeat(3, 1fr); gap: 18px; justify-items: center; align-items: center; margin: 8px 0; max-width: 540px; margin-left: auto; margin-right: auto;">`;
      availableTangrams.forEach((t) => {
        // Disable the tangram if it was selected in the previous trial
        const isDisabled = previous_selection && t === previous_selection;
        html += `
          <div class="tangram-choice${
            isDisabled ? " tangram-disabled" : ""
          }" data-choice="${t}" style="border:2px solid #ccc; border-radius:8px; padding:5px; cursor:${
          isDisabled ? "not-allowed" : "pointer"
        }; width:120px; height:110px; text-align:center;${
          isDisabled ? " pointer-events:none;" : ""
        }">
            <img src="stim/tangrams/tangram_${t}.png" style="width:100px;height:100px;object-fit:contain; display:block; margin:0 auto;" />
          </div>`;
      });
      html += `</div>
        <div style="margin:30px 0; text-align:center;">
          <label for="label-input" style="font-size: 1em; font-weight: bold; margin-bottom: 6px; display: block;"><strong>Write a description for the selected picture:</strong></label>
          <textarea id="label-input" rows="1" cols="30" placeholder="Enter your description here..." required
            style="font-size: 1em; padding: 14px; border-radius: 8px; border: 1.5px solid #8e44ad; width: 80%; max-width: 500px; box-sizing: border-box; margin-bottom: 10px; resize: vertical;"></textarea>
        </div>
        <div style="text-align:center; margin-bottom: 30px;">
          <button id="continue-btn" disabled
            style="font-size: 1em; padding: 14px 40px; background: #8e44ad; color: #fff; border: none; border-radius: 8px; cursor: pointer; transition: background 0.2s; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
            Continue
          </button>
        </div>
        <style>
          .tangram-choice.selected {
            border: 3px solid #8e44ad !important;
          }
          #continue-btn:disabled {
            background: #d2b4e8;
            cursor: not-allowed;
          }
          #continue-btn:not(:disabled):hover {
            background: #6c3483;
          }
          #label-input:focus {
            outline: 2px solid #8e44ad;
            border-color: #6c3483;
          }
        </style>
      `;
      return html;
    },
    choices: "NO_KEYS",
    data: {
      trialInfo: trial,
      trial_num: trial_index,
      task: "tangram_selection",
    },
    on_load: function () {
      let selectedTangram = null;
      const tangramDivs = document.querySelectorAll(".tangram-choice");
      const labelInput = document.getElementById("label-input");
      const continueBtn = document.getElementById("continue-btn");

      tangramDivs.forEach((div) => {
        // Only add event listener if not disabled
        if (!div.classList.contains("tangram-disabled")) {
          div.addEventListener("click", function () {
            tangramDivs.forEach((d) => {
              d.style.border = "2px solid #ccc";
              d.classList.remove("selected");
            });
            this.style.border = "3px solid #8e44ad";
            this.classList.add("selected");
            selectedTangram = this.getAttribute("data-choice");
            checkEnableContinue();
          });
        }
      });

      labelInput.addEventListener("input", checkEnableContinue);

      function checkEnableContinue() {
        if (selectedTangram && labelInput.value.trim().length > 0) {
          continueBtn.disabled = false;
        } else {
          continueBtn.disabled = true;
        }
      }

      continueBtn.addEventListener("click", function () {
        let this_previous_selection = previous_selection;
        previous_selection = selectedTangram; // Update global variable
        
        let selected_tangram_group = null;
        let selected_tangram_earlier_red = null;
        let selected_tangram_earlier_blue = null;
        let selected_tangram_later_red = null;
        let selected_tangram_later_blue = null;

        if (trial.blue_specific_tangrams.includes(selectedTangram)) {
          selected_tangram_group = "blue_specific";
        } else if (trial.shared_tangrams.includes(selectedTangram)) {
          selected_tangram_group = "shared";
        } else if (trial.red_specific_tangrams.includes(selectedTangram)) {
          selected_tangram_group = "red_specific";
        }

        if (selected_tangram_group === "red_specific") {
          selected_tangram_earlier_red = trial.convo_info.red[selectedTangram].earlier_label;
          selected_tangram_later_red = trial.convo_info.red[selectedTangram].label;
        } else if (selected_tangram_group === "blue_specific") {
          selected_tangram_earlier_blue = trial.convo_info.blue[selectedTangram].earlier_label;
          selected_tangram_later_blue = trial.convo_info.blue[selectedTangram].label;
        }

        if (selected_tangram_group === "shared") {
          selected_tangram_earlier_red = trial.convo_info.red[selectedTangram].earlier_label;
          selected_tangram_later_red = trial.convo_info.red[selectedTangram].label;
          selected_tangram_earlier_blue = trial.convo_info.blue[selectedTangram].earlier_label;
          selected_tangram_later_blue = trial.convo_info.blue[selectedTangram].label;
        }

        const trialData = {
          selected_tangram: selectedTangram,
          selected_tangram_group: selected_tangram_group,
          selected_tangram_earlier_red: selected_tangram_earlier_red,
          selected_tangram_earlier_blue: selected_tangram_earlier_blue,
          selected_tangram_later_red: selected_tangram_later_red,
          selected_tangram_later_blue: selected_tangram_later_blue,
          previous_selection: this_previous_selection,
          written_label: labelInput.value.trim(),
        };

        // console.log("Full trial info:", {
        //   trial: trial,
        //   response: trialData,
        // });

        jsPsych.finishTrial(trialData);
      });
    },
  };
}

async function createSelectionTrials(item_id, counterbalance, jsPsych) {
  const trialInfo = await createFreeResponseTrialInfo(
    item_id,
    counterbalance,
    jsPsych
  );
  const selection_phase_timeline = [];

  // Instructions trial
  selection_phase_timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <h2>🤨 Part 2: Questions 🤨</h2>
      <div class="align-left">
      <p>
        In this phase, you will be asked to <strong>choose a picture</strong> and <strong>write a description</strong> for it.
      </p>
      <p>
        On each trial, you will see:
        <ul>
          <li>An <strong>audience</strong> (blue group members and/or naive participants)</li>
          <li>What question the audience will be answering</li>
          <li>12 pictures to choose from</li>
        </ul>
      </p>
      <p>
        <strong>Important:</strong> You must choose a different picture on each trial than you chose on the previous trial.
      </p>
      <p>You will do this ${trialInfo.length} times.</p>
      </div>
    `,
    choices: ["Continue"],
  });

  // Randomize trial order
  const randomizedTrials = jsPsych.randomization.shuffle(trialInfo);

  // Build selection trials, enforcing no repeat tangram
  for (let i = 0; i < randomizedTrials.length; i++) {
    const trial = randomizedTrials[i];
    selection_phase_timeline.push(makeFreeResponseSelectionTrial(trial, i));
    selection_phase_timeline.push({
      type: jsPsychHtmlKeyboardResponse,
      stimulus: "Recording selection...",
      choices: "NO_KEYS",
      trial_duration: function () {
        return jsPsych.randomization.sampleWithoutReplacement(
          [1500, 1750],
          1
        )[0];
      },
    });
  }

  return selection_phase_timeline;
}

function generateFreeResponseStimulus(trial) {
  const svg = generateAudienceSVG(trial.n_blue, trial.n_naive);
  const text = generateFreeResponseStimulusText(trial.goal);
  const count = generateFreeResponseStimulusCount(trial.n_blue, trial.n_naive);

  return `
    <div style="text-align: center;">
      ${svg}
      ${text}
      ${count}
    </div>
  `;
}

function generateFreeResponseStimulusText(goal) {
  if (goal === "refer") {
    return `Your goal is to help <strong>everyone</strong> in the audience <strong>choose the correct picture</strong>.`;
  } else if (goal === "social") {
    return `Your goal is to help the <strong style="color: blue;">blue group</strong> members <strong>identify you as a member of their group</strong>.`;
  }
}

function generateFreeResponseStimulusCount(nBlue, nNaive) {
  const blueText = nBlue === 1 ? "player" : "players";
  const naiveText = nNaive === 1 ? "player" : "players";

  let countText = `${nBlue} ${blueText} in the <strong style="color: blue;">blue group</strong>`;
  if (nNaive > 0) {
    countText += `<br>${nNaive} <strong style="color: gray;">naive ${naiveText}</strong>`;
  }

  return `
    <div style="font-size: 14px; line-height: 1.5; margin: 10px 0;">
      ${countText}
    </div>`;
}
