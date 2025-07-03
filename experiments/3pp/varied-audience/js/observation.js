async function getAvailableTangrams(item_id, counterbalance) {
  const response = await fetch(
    `stim/items/item_${item_id}_${counterbalance}_game_info.json`
  );
  const data = await response.json();

  // get the possible tangrams, which are the keys in the 2nd level
  const available_tangrams = Object.keys(data.blue);

  return available_tangrams;
}

async function createVideoTrials(item_id, counterbalance, jsPsych) {
  try {
    const available_tangrams = await getAvailableTangrams(
      item_id,
      counterbalance
    );

    const video_trials = [];
    const pass_intro = {
      type: jsPsychHtmlButtonResponse,
      stimulus: `<p>Congrats on passing the comprehension check!</p><p>Please press "continue" to begin the first part of the task.</p>`,
      choices: ["Continue"],
    };
    video_trials.push(pass_intro);

    for (let repNum = 0; repNum < 6; repNum++) {
      for (let color of ["blue"]) {
        // randomize the order of the tangrams
        const prep_trial = {
          type: jsPsychHtmlButtonResponse,
          stimulus: `
          <h2>Round ${repNum + 1}</h2>`,
          choices: ["Continue"],
        };

        video_trials.push(prep_trial);

        const shuffled_tangrams =
          jsPsych.randomization.shuffle(available_tangrams);
        for (let tangram of shuffled_tangrams) {
          const video_trial = {
            type: jsPsychVideoButtonResponse,
            stimulus: [
              `stim/convo_vids/videos/480p15/item_${item_id}_${counterbalance}_${color}_target_${tangram}_repNum_${repNum}.mp4`,
            ],
            width: 800,
            choices: ["Continue"],
            prompt: `
            <h2><span style="color: ${color}">${
              color.charAt(0).toUpperCase() + color.slice(1)
            }</span> group</h2>
            <h3>Round ${repNum + 1}</h3>
            <p>Please press continue when you are finished.</p>
            `,
            controls: false,
            autoplay: true,
            response_allowed_while_playing: false,
          };
          video_trials.push(video_trial);
        }
      }
      const end_of_round = {
        type: jsPsychHtmlButtonResponse,
        stimulus: `<h1>End of round ${repNum + 1}</h1>
        <p>Please press continue${
          repNum < 5 ? " to move on to the next round" : ""
        }.</p>`,
        choices: ["Continue"],
      };
      video_trials.push(end_of_round);
    }

    const end_of_video_trials = {
      type: jsPsychHtmlButtonResponse,
      stimulus: `<h1>End of first phase</h1>
      <p>Please press continue to move on to the second phase.</p>`,
      choices: ["Continue"],
    };
    video_trials.push(end_of_video_trials);

    return video_trials;
  } catch (error) {
    console.error("Error loading video trials:", error);
    throw error;
  }
}
