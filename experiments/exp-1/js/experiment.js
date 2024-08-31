// Main experiment file

const jsPsych = initJsPsych({ show_progress_bar: true });

var timeline = [];

makeTrials(0, 1, jsPsych).then((trials) => {
  timeline.push(trials);
  jsPsych.run(timeline.flat());
});
