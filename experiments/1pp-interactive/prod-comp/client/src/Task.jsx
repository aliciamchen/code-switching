import {
  useGame,
  usePlayer,
  usePlayers,
  useRound,
  useStage,
} from "@empirica/core/player/classic/react";
import { Loading } from "@empirica/core/player/react";
import React from "react";
import { Tangram } from "./components/Tangram.jsx";

export function Task() {
  const game = useGame();
  const player = usePlayer();
  const players = usePlayers();
  const round = useRound();
  const stage = useStage();
  const target = stage.get("name") == "Selection"? stage.get("target") : round.get("lastTarget");  // TODO: this doesnt exist in the stage
  const shuffled_tangrams = player.get("shuffled_tangrams");
  const lastCorrect = player.round.get("lastClicked") == target;
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
  // let feedback = (
  //   player.get('clicked') == '' ? '' :
  //     correct ? "Correct! You earned 3 points!" :
  //     "Ooops, that wasn't the target! You earned no bonus this round."
  // )
  let feedback = "";
  if (stage.get("name") == "Feedback") {
    if (player.round.get("role") == "listener") {
      // if player did not respond in time
      if (!player.round.get("lastClicked")) { // this should be instead based on whether they submitted in the PREVIOUS stage
        feedback =
          "You did not respond in time. You earned no bonus this round.";
      } else if (lastCorrect) {
        feedback = "Correct! You earned one point";
      } else {
        feedback =
          "Ooops, that wasn't the target! You earned no bonus this round.";  // TODO: this shows up even if its correct
      }
    }
    if (player.round.get("role") == "speaker") {
        feedback = `You earned ${player.round.get("last_stage_score")} points this round.`;  // stsage score should be based on PREVIOUS stage.. switch to round score? 
    }
  }
  return (
    <div className="task">
      <div className="board">
        <div className="header" style={{ display: "flex" }}>
          <h1
            className="roleIndicator"
            style={{ float: "left", marginLeft: "50px" }}
          >
            {" "}
            You are the {player.round.get("role")}.
          </h1>
          <h3
            className="feedbackIndicator"
            style={{
              float: "left",
              marginLeft: "50px",
              marginTop: "auto",
              marginBottom: "auto",
            }}
          >
            <>{feedback}</>
          </h3>
        </div>
        <div className="all-tangrams">
          <div className="tangrams grid">{tangramsToRender}</div>
        </div>
      </div>
    </div>
  );
}
