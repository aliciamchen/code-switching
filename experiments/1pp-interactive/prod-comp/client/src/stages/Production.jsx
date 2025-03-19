import {
  useGame,
  usePlayer,
  usePlayers,
  useRound,
  useStage,
} from "@empirica/core/player/classic/react";
import React from "react";
import { Tangram } from "../components/Tangram.jsx";

// Participants are asked to
// generate the best labels for inducing a particular audience (a member of their ‘own’ group or a member of the
//     ‘other’ group) to make one of two inferences (the ‘referential’ goal of choosing which tangram is being referred
//     to, or the ‘social’ goal of being identified as a member of the audience group).
export function Production() {
  const game = useGame();
  const player = usePlayer();
  const players = usePlayers();
  const round = useRound();
  const stage = useStage();

  const target = player.stage.get("target");
  const condition = player.stage.get("condition");

  let prompt;
  if (condition == "refer own") {
    prompt =
      "Please write a description to help a member of your own group pick the correct picture.";
  } else if (condition == "refer other") {
    prompt =
      "Please write a description to help a member of the other group pick the correct picture.";
  } else if (condition == "social own") {
    prompt =
      "Please write a description of the target picture, to help a member of your own group identify you as a member of the group.";
  }

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

  return (
    <div className="task">
      <div className="board">
        <div className="prompt-container">
          <p className="instruction-prompt">{prompt}</p>
        </div>
      </div>
      <div className="all-tangrams">
        <div className="tangrams grid">{tangramsToRender}</div>
      </div>
    </div>
  );
}
