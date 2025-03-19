import { useRound } from "@empirica/core/player/classic/react";
import { Loading } from "@empirica/core/player/react";
import React from "react";
import { Refgame } from "./stages/Refgame.jsx";

export function Task() {
  const round = useRound();
  switch (round.get("phase")) {
    case "refgame":
      return <Refgame />;
    default:
      return <Loading />;
  }
}
