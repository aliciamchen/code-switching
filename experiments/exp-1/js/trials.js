// put all the trials together
async function makeTrials(item_id, participant_id, jsPsych) {
  try {
    const timeline = [];

    // consent
    const consent = await createConsent();
    timeline.push(consent);

    // comprehension check loop
    const comprehensionLoop = await createComprehensionLoop(jsPsych);
    timeline.push(comprehensionLoop);

    // TODO: observation phase

    // selection phase
    const selectionTrials = await createSelectionTrials(item_id, participant_id, jsPsych);
    timeline.push(selectionTrials);

    // exit survey
    const exitSurvey = await createExitSurvey();
    timeline.push(exitSurvey);

    return timeline.flat();
  } catch (error) {
    console.error("Error loading trials:", error);
    throw error;
  }
}
