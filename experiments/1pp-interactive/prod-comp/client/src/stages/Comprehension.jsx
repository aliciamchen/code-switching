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
  return "Not implemented yet";
  // if player.stage.get(clicked), and group is selected, then submit
}
