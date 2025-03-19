import { useRound } from "@empirica/core/player/classic/react";
import { Loading } from "@empirica/core/player/react";
import React from "react";
import { Refgame } from "./stages/Refgame.jsx";
import { Production } from "./stages/Production.jsx";
import { Comprehension } from "./stages/Comprehension.jsx";

export function Task() {
  const round = useRound();
  switch (round.get("phase")) {
    case "refgame":
      return <Refgame />;
    case "speaker_prod": 
        return <Production />;
    case "comprehension": 
        return <Comprehension />;
    default:
      return <Loading />;
  }
}
