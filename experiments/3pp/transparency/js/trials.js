// put all the trials together
async function makeTrials(jsPsych) {
  try {

    jsPsych.data.addProperties({
      subject_id: subject_id,
      study_id: study_id,
      session_id: session_id,
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

    // instructions
    const instructions = await createInstructions();
    timeline.push(instructions);

    // selection phase
    const selectionTrials = await createSelectionTrials(jsPsych);
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
