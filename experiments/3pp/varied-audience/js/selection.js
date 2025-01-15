// Selection phase

// Filter the relevant trials from the JSON file
async function loadTrialInfo(item_id, counterbalance, jsPsych) {
  const response = await fetch(
    `stim/2AFC_trials/item_${item_id}_${counterbalance}_2AFC.json`
  );
  const trials = await response.json();
  const baseline_trials = trials.filter((trial) => trial.type === "baseline");
  const main_trials_all = trials.filter((trial) => trial.type === "main");

  const ingroupLevels = [0, 1, 2, 3, 4];
  const outgroupLevels = [0, 1, 2, 4, 8, 16];
  const allAudienceConditions = [];

  // create all possible audience conditions
  ingroupLevels.forEach((ingroup) => {
    outgroupLevels.forEach((outgroup) => {
      if (ingroup === 0 && outgroup === 0) return;
      allAudienceConditions.push({ n_ingroup: ingroup, n_outgroup: outgroup });
    });
  });

  // assign a tangram to each condition, with the constraint that 5 tangrams appear 5x, 1 tangram appears 4x
  const tangrams = new Set();
  main_trials_all.forEach((trial) => {
    tangrams.add(trial.tangram);
  }); // this is probably slow...
  const tangramArray = Array.from(tangrams);
  // duplicate the tangrams array 5 times and get rid of a random value
  const tangramAssignments = Array.from(
    { length: 5 },
    () => tangramArray
  ).flat();
  const randomIndex = Math.floor(Math.random() * tangramAssignments.length);
  tangramAssignments.splice(randomIndex, 1);
  // shuffle
  const shuffledAssignments = jsPsych.randomization.shuffle(tangramAssignments);

  // assign tangrams to audience conditions
  allAudienceConditions.forEach((audience, i) => {
    audience.tangram = shuffledAssignments[i];
  });

  // create refer and social conditions
  const referConditions = allAudienceConditions.map((audience) => {
    audience.goal = "refer";
    return audience;
  });
  const socialConditions = allAudienceConditions
    .filter((audience) => audience.n_ingroup > 0)
    .map((audience) => {
      audience.goal = "social";
      return audience;
    });

  const allConditions = referConditions.concat(socialConditions);
  // how many conditions in total? should be 53
  console.assert(
    allConditions.length === 53,
    `Expected 53 but got ${allConditions.length}`
  );

  // create the trials
  const main_trials = [];
  allConditions.forEach((condition) => {
    const matchingMainTrial = main_trials_all.find(
      (trial) =>
        trial.n_ingroup === condition.n_ingroup &&
        trial.n_outgroup === condition.n_outgroup &&
        trial.goal === condition.goal &&
        trial.tangram === condition.tangram
    );
    if (matchingMainTrial) {
      main_trials.push(matchingMainTrial);
    } else {
      console.error("No matching trial for condition:", condition);
    }
  });
  const all_trials = main_trials.concat(baseline_trials);
  return all_trials;
}

async function createSelectionTrials(item_id, counterbalance, jsPsych) {
  try {
    const trialInfo = await loadTrialInfo(item_id, counterbalance, jsPsych);
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
        On each screen, you will see two pictures with a word describing each, and
        you will have to pick one of them. We will tell you
        <strong>who</strong> will see your choice (a number of players from the game
        you observed, and a number of players who did not play this game) and <strong>what question</strong> they will be answering when they
        see it. Your goal is to try to get them to answer the question in a certain
        way based on the label you select.
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
