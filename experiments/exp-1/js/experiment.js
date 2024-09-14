// Main experiment file

const local_testing = false;

const params = {
  perTrialBonus: 0.25,
  n_tangrams: 6
}

// TODO: get item id and participant id from datapipe condition assignment
var item_id = 0; // which item (set of stimuli?)
var participant_id = 5; // which participant (determines what 2AFC set)

const jsPsych = initJsPsych({
  on_finish: function () {
    if (local_testing) {
      jsPsych.data.displayData();
      jsPsych.data.get().localSave("json", "testdata.json");
    }
  },
  show_progress_bar: true,
});


var subject_id =
  local_testing || jsPsych.data.getURLVariable("PROLIFIC_PID") == undefined
    ? jsPsych.randomization.randomID(12)
    : jsPsych.data.getURLVariable("PROLIFIC_PID");
console.log("subject_id:", subject_id);
var study_id = jsPsych.data.getURLVariable("STUDY_ID");
var session_id = jsPsych.data.getURLVariable("SESSION_ID");

jsPsych.data.addProperties({
  subject_id: subject_id,
  study_id: study_id,
  session_id: session_id,
  item_id: item_id,
  url: window.location.href,
});

var timeline = [];

makeTrials(item_id, participant_id, jsPsych).then((trials) => {
  timeline.push(trials);
  jsPsych.run(timeline.flat());
});
