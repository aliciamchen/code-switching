// Main experiment file

const jsPsych = initJsPsych();

const video_trial = {
  type: jsPsychVideoButtonResponse,
  stimulus: ["stim/videos/test.mp4"],
  width: 800,
  choices: ["Continue"],
  prompt: "<p>Please press continue when you are finished.</p>",
  controls: true,
  autoplay: false,
  response_allowed_while_playing: false,
};

var timeline = [];

makeTrials(1, 1, jsPsych).then((trials) => {
    timeline.push(trials);
    jsPsych.run(timeline.flat());
});



