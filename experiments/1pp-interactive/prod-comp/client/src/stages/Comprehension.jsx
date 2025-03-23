import React, { useState, useEffect } from "react";
import { Tangram } from "../components/Tangram.jsx";

export function Comprehension(props) {
  const { round, stage, game, player, players } = props;
  const target = player.stage.get("target");

  // participnats can freely select the tangram before submitting
  const [clickedTangram, setClickedTangram] = useState(
    player.stage.get("clicked_tangram")
  );
  useEffect(() => {
    setClickedTangram(player.stage.get("clicked_tangram"));
  }, [player.stage.get("clicked_tangram")]);

  const handleTangramClick = (tangram) => {
    player.stage.set("clicked_tangram", tangram);
    setClickedTangram(tangram);
  };

  // and freely select the group
  const [clickedGroup, setClickedGroup] = useState(null);
  const handleGroupClick = (group) => {
    setClickedGroup(group);
  };

  // if both are selected, then submit
  const handleSubmit = () => {
    if (clickedTangram && clickedGroup) {
      player.stage.set("clicked_tangram", clickedTangram);
      player.stage.set("clicked_group", clickedGroup);
      player.stage.set("submit", true);
    }
  };

  const shuffled_tangrams = player.get("shuffled_tangrams");
  let tangramsToRender;
  if (shuffled_tangrams) {
    tangramsToRender = shuffled_tangrams.map((tangram, i) => (
      <Tangram
        key={tangram}
        tangram={tangram}
        tangram_num={i}
        round={round}
        stage={stage}
        game={game}
        player={player}
        players={players}
        target={target}
        onSelect={handleTangramClick}
      />
    ));
  }

  // Define the prompt
  const prompt = (
    <div>
      <p className="instruction-prompt">
        Speaker's description:{" "}
        <span
          style={{
            fontStyle: "italic",
            backgroundColor: "#f5f5f5",
            padding: "4px 8px",
            borderRadius: "4px",
            marginLeft: "8px",
            display: "inline-block",
          }}
        >
          {player.stage.get("description")}
        </span>
      </p>
      <p className="instruction-prompt" style={{ marginTop: "15px" }}>
        Please select which picture you think they were describing.
      </p>
    </div>
  );

  if (player.stage.get("submit")) {
    return (
      <div className="task">
        <div className="board">
          <div className="prompt-container">
            <p className="instruction-prompt">Waiting for other players...</p>
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="task">
      <div className="board">
        <div className="trial-info">
          Trial {stage.get("trial_num") + 1} out of 36
        </div>
        <div className="prompt-container">{prompt}</div>
      </div>
      <div className="all-tangrams">
        <div className="tangrams grid">{tangramsToRender}</div>
      </div>
    </div>
  );
  // if player.stage.get(clicked), and group is selected, then submit
}
