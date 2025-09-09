async function getAvailableTangrams(item_id, counterbalance) {
  const response = await fetch(
    `stim/items/item_${item_id}_${counterbalance}_game_info.json`
  );
  const data = await response.json();

  // get all tangrams from both groups
  const blue_tangrams = Object.keys(data.blue);
  const red_tangrams = Object.keys(data.red);

  // combine all unique tangrams
  const all_tangrams = [...new Set([...blue_tangrams, ...red_tangrams])];

  return { blue_tangrams, red_tangrams, all_tangrams, data };
}

async function createVideoTrials(item_id, counterbalance, jsPsych) {
  try {
    const { blue_tangrams, red_tangrams, all_tangrams, data } =
      await getAvailableTangrams(item_id, counterbalance);

    const video_trials = [];

    // Initial instructions
    const initial_instructions = {
      type: jsPsychHtmlButtonResponse,
      stimulus: `
          <p>Congrats on passing the comprehension check!</p><p>Please press "continue" to begin the first part of the task.</p>
      `,
      choices: ["Continue"],
    };
    video_trials.push(initial_instructions);

    // Phase 1: First round for all players (both groups)
    const phase1_intro = {
      type: jsPsychHtmlButtonResponse,
      stimulus: `
      <h2>Phase 1: First Round</h2>
      <p>You will now see the first round where both groups talk about their pictures.</p>
      <p>The red group will talk about 8 pictures, and the blue group will talk about 8 pictures.</p>
      `,
      choices: ["Continue"],
    };
    video_trials.push(phase1_intro);

    // Show first round for red group
    const red_first_round = {
      type: jsPsychHtmlButtonResponse,
      stimulus: `
      <h2>Red Group - Round 1</h2>
      <p>You will now see the red group talk about their pictures.</p>
      `,
      choices: ["Continue"],
    };
    video_trials.push(red_first_round);

    const shuffled_red_tangrams = jsPsych.randomization.shuffle(red_tangrams);
    for (let tangram of shuffled_red_tangrams) {
      const video_trial = {
        type: jsPsychVideoButtonResponse,
        stimulus: [
          `stim/convo_vids/videos/480p15/item_${item_id}_${counterbalance}_red_target_${tangram}_repNum_0.mp4`,
        ],
        width: 800,
        choices: ["Continue"],
        prompt: `
        <h2><span style="color: red;">Red</span> group</h2>
        <h3>Round 1</h3>
        <p>Please press continue when you are finished.</p>
        `,
        controls: false,
        autoplay: true,
        response_allowed_while_playing: false,
      };
      video_trials.push(video_trial);
    }

    // Show first round for blue group
    const blue_first_round = {
      type: jsPsychHtmlButtonResponse,
      stimulus: `
      <h2>Blue Group - Round 1</h2>
      <p>You will now see the blue group talk about their pictures.</p>
      `,
      choices: ["Continue"],
    };
    video_trials.push(blue_first_round);

    const shuffled_blue_tangrams = jsPsych.randomization.shuffle(blue_tangrams);
    for (let tangram of shuffled_blue_tangrams) {
      const video_trial = {
        type: jsPsychVideoButtonResponse,
        stimulus: [
          `stim/convo_vids/videos/480p15/item_${item_id}_${counterbalance}_blue_target_${tangram}_repNum_0.mp4`,
        ],
        width: 800,
        choices: ["Continue"],
        prompt: `
        <h2><span style="color: blue;">Blue</span> group</h2>
        <h3>Round 1</h3>
        <p>Please press continue when you are finished.</p>
        `,
        controls: false,
        autoplay: true,
        response_allowed_while_playing: false,
      };
      video_trials.push(video_trial);
    }

    // Phase 2: Focus on blue group convergence
    const phase2_intro = {
      type: jsPsychHtmlButtonResponse,
      stimulus: `
      <h2>Phase 2: Blue Group</h2>
      <p>Now we will focus on the blue group as they develop their communication system.</p>
      <p>You will see them talk about their pictures over multiple rounds.</p>
      `,
      choices: ["Continue"],
    };
    video_trials.push(phase2_intro);

    // Show blue group convergence (rounds 2-4)
    for (let repNum = 1; repNum < 5; repNum++) {
      const round_intro = {
        type: jsPsychHtmlButtonResponse,
        stimulus: `
        <h2>Blue Group - Round ${repNum + 1}</h2>
        `,
        choices: ["Continue"],
      };
      video_trials.push(round_intro);

      const shuffled_blue_tangrams_round =
        jsPsych.randomization.shuffle(blue_tangrams);
      for (let tangram of shuffled_blue_tangrams_round) {
        const video_trial = {
          type: jsPsychVideoButtonResponse,
          stimulus: [
            `stim/convo_vids/videos/480p15/item_${item_id}_${counterbalance}_blue_target_${tangram}_repNum_${repNum}.mp4`,
          ],
          width: 800,
          choices: ["Continue"],
          prompt: `
          <h2><span style="color: blue;">Blue</span> group</h2>
          <h3>Round ${repNum + 1}</h3>
          <p>Please press continue when you are finished.</p>
          `,
          controls: false,
          autoplay: true,
          response_allowed_while_playing: false,
        };
        video_trials.push(video_trial);
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

    // Phase 3: Final round for all players
    const phase3_intro = {
      type: jsPsychHtmlButtonResponse,
      stimulus: `
      <h2>Phase 3: Final Round</h2>
      <p>Now you will see the final round for both groups.</p>
      `,
      choices: ["Continue"],
    };
    video_trials.push(phase3_intro);

    // Show final round for red group
    const red_final_round = {
      type: jsPsychHtmlButtonResponse,
      stimulus: `
      <h2>Red Group - Final Round</h2>
      <p>You will now see the red group's final communication system.</p>
      `,
      choices: ["Continue"],
    };
    video_trials.push(red_final_round);

    const shuffled_red_tangrams_final =
      jsPsych.randomization.shuffle(red_tangrams);
    for (let tangram of shuffled_red_tangrams_final) {
      const video_trial = {
        type: jsPsychVideoButtonResponse,
        stimulus: [
          `stim/convo_vids/videos/480p15/item_${item_id}_${counterbalance}_red_target_${tangram}_repNum_5.mp4`,
        ],
        width: 800,
        choices: ["Continue"],
        prompt: `
        <h2><span style="color: red;">Red</span> group</h2>
        <h3>Final Round</h3>
        <p>Please press continue when you are finished.</p>
        `,
        controls: false,
        autoplay: true,
        response_allowed_while_playing: false,
      };
      video_trials.push(video_trial);
    }

    // Show final round for blue group
    const blue_final_round = {
      type: jsPsychHtmlButtonResponse,
      stimulus: `
      <h2>Blue Group - Final Round</h2>
      <p>You will now see the blue group's final communication system.</p>
      `,
      choices: ["Continue"],
    };
    video_trials.push(blue_final_round);

    const shuffled_blue_tangrams_final =
      jsPsych.randomization.shuffle(blue_tangrams);
    for (let tangram of shuffled_blue_tangrams_final) {
      const video_trial = {
        type: jsPsychVideoButtonResponse,
        stimulus: [
          `stim/convo_vids/videos/480p15/item_${item_id}_${counterbalance}_blue_target_${tangram}_repNum_5.mp4`,
        ],
        width: 800,
        choices: ["Continue"],
        prompt: `
        <h2><span style="color: blue;">Blue</span> group</h2>
        <h3>Final Round</h3>
        <p>Please press continue when you are finished.</p>
        `,
        controls: false,
        autoplay: true,
        response_allowed_while_playing: false,
      };
      video_trials.push(video_trial);
    }

    const end_of_video_trials = {
      type: jsPsychHtmlButtonResponse,
      stimulus: `<h1>End of Observation Phase</h1>
      <p>You have now observed both groups' communication systems.</p>
      <p>Please press continue to move on to the next phase.</p>`,
      choices: ["Continue"],
    };
    video_trials.push(end_of_video_trials);

    return video_trials;
  } catch (error) {
    console.error("Error loading video trials:", error);
    throw error;
  }
}
