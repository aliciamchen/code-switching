async function loadInstructionsFromHTML() {
  try {
    const response = await fetch("html/instructions.html");
    const text = await response.text();
    return text.split("<!-- PAGE BREAK -->");
  } catch (error) {
    console.error("Error loading instructions:", error);
    throw error;
  }
}

async function createInstructions() {
    try {
        const instructionPages = await loadInstructionsFromHTML();
        return {
        type: jsPsychInstructions,
        pages: instructionPages,
        show_clickable_nav: true
        };
    } catch (error) {
        console.error("Error creating instructions:", error);
        throw error;
    }
}