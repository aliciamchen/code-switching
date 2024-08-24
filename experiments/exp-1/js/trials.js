// put all the trials together
async function makeTrials(item_id, participant_id, jsPsych) {
  try {
    const timeline = [];

    // instructions
    const instructions = await createInstructions();
    timeline.push(instructions);

    // TODO: observation phase

    // selection phase
    const selectionTrials = await createSelectionTrials(item_id, participant_id, jsPsych);
    timeline.push(selectionTrials);

    return timeline.flat();
  } catch (error) {
    console.error("Error loading trials:", error);
    throw error;
  }
}
