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
        return `<p>Please select a description.</p>\
        <p>Imagine that your chosen description will be shown to <strong>a random player in <em>either</em> group.</strong></p>\
        <p>The question they will answer: <em>"What picture is this person referring to?"</em> <br><strong>You want them to correctly choose the corresponding picture out of the 6 pictures.</strong></p>
        <p></p>`;
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

  var shuffled_options = jsPsych.randomization.repeat(trial.options, 1);

  return [
    {
      type: jsPsychHtmlButtonResponse,
      stimulus: make_stimulus(trial),
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
        In the second part of the task, it will be <strong>up to you</strong> to
        choose words that go with these pictures.
      </p>
      <p>
        On each screen, you will see two pictures with a word describing each, and
        you will have to pick one of them. We will tell you
        <strong>who</strong> will see your choice (a member of the red group, a member of the
        blue group, or a member of either group) and <strong>what question</strong> they will be answering when
        they see it. Your goal is to try to get them to answer the question in a
        certain way based on the word you select.
      </p>
      <p>
        Please read carefully, so you know <strong>who</strong> will see the picture
        and words you chose, <strong>what question</strong> they will be answering,
        and <strong>how</strong> you want them to answer it.
      </p>
      <p>You will do this 69 times. You will receive a bonus of $0.10 for each correct choice, for a total bonus of up to $7.</p>
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
