// Main experiment file

const jsPsych = initJsPsych();

var timeline = [];

makeTrials(0, 1, jsPsych).then((trials) => {
    timeline.push(trials);
    jsPsych.run(timeline.flat());
});



