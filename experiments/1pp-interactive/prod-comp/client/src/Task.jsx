import { Loading } from "@empirica/core/player/react";
import React from "react";
import { Refgame } from "./stages/Refgame.jsx";
import { Production } from "./stages/Production.jsx";
import { Comprehension } from "./stages/Comprehension.jsx";

export function Task(props) {
  const {round, stage, game, player, players} = props;
  switch (round.get("phase")) {
    case "refgame":
      return <Refgame round={round} stage={stage} game={game} player={player} players={players}/>;
    case "speaker_prod": 
        return <Production />;
    case "comprehension": 
        return <Comprehension />;
    default:
      return <Loading />;
  }
}
