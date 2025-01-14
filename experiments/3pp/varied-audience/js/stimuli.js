// Function to generate SVG audience visualization
function generateAudienceSVG(nIngroup, nOutgroup) {
  const avatarWidth = 64;
  const avatarHeight = 56;
  const minDist = 70;

  const totalAvatars = nIngroup + nOutgroup;
  const circleRadius = 25 * Math.sqrt(totalAvatars);

  const centerX = 400;
  const centerY = circleRadius + 50; //150;

  // Helper functions
  function randomPointInCircle(cx, cy, r) {
    const angle = Math.random() * 2 * Math.PI;
    const radius = Math.sqrt(Math.random()) * r;
    const x = cx + radius * Math.cos(angle);
    const y = cy + radius * Math.sin(angle);
    return { x, y };
  }

  function resolveOverlaps(data, cx, cy, r, maxIterations = 1000) {
    let iterations = 0;
    let overlaps = true;

    while (overlaps && iterations < maxIterations) {
      overlaps = false;
      iterations++;

      for (let i = 0; i < data.length; i++) {
        for (let j = i + 1; j < data.length; j++) {
          const dx = data[j].x - data[i].x;
          const dy = data[j].y - data[i].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < minDist) {
            overlaps = true;

            const overlap = minDist - dist;
            const angle = Math.atan2(dy, dx);
            const moveX = (overlap / 2) * Math.cos(angle);
            const moveY = (overlap / 2) * Math.sin(angle);

            data[i].x -= moveX;
            data[i].y -= moveY;
            data[j].x += moveX;
            data[j].y += moveY;

            [data[i], data[j]].forEach((d) => {
              const dx = d.x - cx;
              const dy = d.y - cy;
              const distFromCenter = Math.sqrt(dx * dx + dy * dy);
              if (distFromCenter > r) {
                const shrinkFactor = r / distFromCenter;
                d.x = cx + dx * shrinkFactor;
                d.y = cy + dy * shrinkFactor;
              }
            });
          }
        }
      }
    }
  }

  // Initialize avatars
  const data = [];
  for (let i = 0; i < nIngroup; i++) {
    data.push({
      ...randomPointInCircle(centerX, centerY, circleRadius),
      color: "blue",
    });
  }
  for (let i = 0; i < nOutgroup; i++) {
    data.push({
      ...randomPointInCircle(centerX, centerY, circleRadius),
      color: "gray",
    });
  }

  resolveOverlaps(data, centerX, centerY, circleRadius);

  // Generate SVG as string
  let svgContent = `<svg id="audience" width="800" height="${
    2 * circleRadius + 100
  }" xmlns="http://www.w3.org/2000/svg">`;
  data.forEach((d) => {
    const shiftX = d.x - avatarWidth / 2;
    const shiftY = d.y - avatarHeight / 2;
    svgContent += `
        <g transform="translate(${shiftX},${shiftY})">
          <ellipse cx="32.27" cy="22.37" rx="7.43" ry="7.35" fill="${d.color}"></ellipse>
          <path d="M42.67,38.23c-1.27,2.89-9.94,2.9-10.55,2.9-.88,0-9.01-.11-10.24-2.9-1.19-2.69,4.16-7.6,10.4-7.6,6.23,0,11.58,4.9,10.4,7.6Z" fill="${d.color}"></path>
        </g>`;
  });
  svgContent += `</svg>`;
  return svgContent;
}

function generateStimulusText(goal) {
  if (goal === "refer") {
    return `<p>Please select the best description for the players in the audience to <strong>choose the correct picture.</strong></p>`;
  } else if (goal === "social") {
    return `<p>Please select the best description for the players in the audience to <strong>identify you as a member of their group.</strong></p>`;
  }
}

function generateStimulusCount(nIngroup, nOutgroup) {
  const ingroupText = nIngroup === 1 ? "player" : "players";
  const outgroupText = nOutgroup === 1 ? "player" : "players";
  return `
    <div style="font-size: 12px; line-height: 1.5;">
        ${nIngroup} ${ingroupText} in the <strong style="color: blue;">blue group</strong>
        <br>${nOutgroup} ${outgroupText} in <strong style="color: gray;">other groups</strong>
    </div>`;
}

function generateStimulus(trial) {
  const svg = generateAudienceSVG(trial.n_ingroup, trial.n_outgroup);
  const text = generateStimulusText(trial.goal);
  const count = generateStimulusCount(trial.n_ingroup, trial.n_outgroup);
  return `${text}${svg}${count}`;
}
