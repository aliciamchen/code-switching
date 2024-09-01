// Main experiment file

const local_testing = true;

const params = {
  perTrialBonus: 0.5,
  n_tangrams: 6
}

const jsPsych = initJsPsych({
  on_finish: function () {
    jsPsych.data.displayData();
    if (local_testing) {
      jsPsych.data.get().localSave("json", "testdata.json");
    }
  },
  show_progress_bar: true,
});

var item_id = 0; // which item (set of stimuli?)
var participant_id = 5; // which participant (determines what 2AFC set)

var subject_id =
  local_testing || jsPsych.data.getURLVariable("PROLIFIC_PID") == ""
    ? jsPsych.randomization.randomID(12)
    : jsPsych.data.getURLVariable("PROLIFIC_PID");
var study_id = jsPsych.data.getURLVariable("STUDY_ID");
var session_id = jsPsych.data.getURLVariable("SESSION_ID");

jsPsych.data.addProperties({
  subject_id: subject_id,
  study_id: study_id,
  session_id: session_id,
  item_id: item_id,
  participant_id: participant_id,
  url: window.location.href,
});

var timeline = [];

makeTrials(item_id, participant_id, jsPsych).then((trials) => {
  timeline.push(trials);
  jsPsych.run(timeline.flat());
});
