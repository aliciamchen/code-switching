// Consent
async function createConsent() {
  try {
    const response = await fetch("html/consent.html");
    const text = await response.text();
    return {
      type: jsPsychHtmlButtonResponse,
      stimulus: text,
      choices: ["I consent to participate"],
    };
  } catch (error) {
    console.error("Error creating consent:", error);
    throw error;
  }
}

// Instructions and comprehension loop
async function loadInstructionsFromHTML() {
  try {
    const response = await fetch("html/instructions.html");
    let text = await response.text();
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
      show_clickable_nav: true,
      show_page_number: true,
    };
  } catch (error) {
    console.error("Error creating instructions:", error);
    throw error;
  }
}
