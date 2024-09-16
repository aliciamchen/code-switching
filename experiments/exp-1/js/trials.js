// put all the trials together
async function makeTrials(jsPsych) {
  try {
    const condition = await jsPsychPipe.getCondition("ZycTOU10DI0v");
    const item_id = condition % 2;
    console.log(condition, item_id);

    jsPsych.data.addProperties({
      subject_id: subject_id,
      study_id: study_id,
      session_id: session_id,
      condition: condition,
      item_id: item_id,
      url: window.location.href,
    });

    const timeline = [];

    // preload all of the videos in stim/convo_vids/videos/480p15 if they correspond to the item_id
    const preload = {
      type: jsPsychPreload,
      auto_preload: true,
    };
    timeline.push(preload);

    // consent
    const consent = await createConsent();
    timeline.push(consent);

    // instructions + comprehension check loop
    const comprehensionLoop = await createComprehensionLoop(jsPsych);
    timeline.push(comprehensionLoop);

    // observation phase
    const videoTrials = await createVideoTrials(item_id, jsPsych);
    timeline.push(videoTrials);

    // selection phase
    const selectionTrials = await createSelectionTrials(item_id, jsPsych);
    timeline.push(selectionTrials);

    // exit survey
    const exit = await createExit(jsPsych);
    timeline.push(exit);

    return timeline.flat();
  } catch (error) {
    console.error("Error loading trials:", error);
    throw error;
  }
}
