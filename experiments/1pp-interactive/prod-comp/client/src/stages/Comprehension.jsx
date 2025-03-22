import React from "react";
import { Tangram } from "../components/Tangram.jsx";

export function Comprehension(props) {
  const { round, stage, game, player, players } = props;
  const target = player.stage.get("target");
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
      />
    ));
  }

   // Define the prompt
   const prompt = (
    <>
      <p>
        The speaker described this picture as:
      </p>
      <p className="description">{player.stage.get("description")}</p>
      <p>Please select which picture you think they were describing.</p>
    </>
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
        <div className="prompt-container">
          {prompt}
        </div>
      </div>
      <div className="all-tangrams">
        <div className="tangrams grid">{tangramsToRender}</div>
      </div>
    </div>
  );
  // if player.stage.get(clicked), and group is selected, then submit
}
