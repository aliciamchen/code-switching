import {
  useGame,
  usePlayer,
  usePlayers,
  useRound,
  useStage,
} from "@empirica/core/player/classic/react";
import React from "react";
import { Tangram } from "../components/Tangram.jsx";

export function Refgame() {
  const game = useGame();
  const player = usePlayer();
  const players = usePlayers();
  const round = useRound();
  const stage = useStage();

  const target = round.get("target");
  const shuffled_tangrams = player.get("shuffled_tangrams");
  const correct = player.round.get("clicked") == target;
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

  let feedback = "";
  if (stage.get("name") == "Feedback") {
    if (player.round.get("role") == "listener") {
      // if player did not respond in time
      if (!player.round.get("clicked")) {
        // this should be instead based on whether they submitted in the PREVIOUS stage
        feedback =
          "You did not respond in time. You earned no bonus this round.";
      } else if (correct) {
        feedback = "Correct! You earned three points.";
      } else {
        feedback =
          "Ooops, that wasn't the target! You earned no bonus this round.";
      }
    }
    if (player.round.get("role") == "speaker") {
      feedback = `You earned ${player.round.get(
        "round_score"
      )} points this round.`;
    }
  }
  return (
    <div className="task">
      <div className="board">
        <h1 className="roleIndicator" style={{ textAlign: "center" }}>
          {" "}
          {player.round.get("role") == "speaker"
            ? "You are the speaker. Please describe the picture in the box to the other players."
            : "You are a listener. Please click on the image that the speaker describes."}
        </h1>

        <div className="all-tangrams">
          <div className="tangrams grid">{tangramsToRender}</div>
        </div>
        <h3
          className="feedbackIndicator"
          style={{
            marginTop: 5,
            marginBottom: "auto",
            textAlign: "center",
            fontWeight: "bold",
            width: "100%",
          }}
        >
          <>{feedback}</>
        </h3>
      </div>
    </div>
  );
}
