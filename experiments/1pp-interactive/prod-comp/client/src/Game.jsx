import {
  Chat,
  usePlayer,
  useGame,
  useStage,
} from "@empirica/core/player/classic/react";

import React from "react";
import { Profile } from "./Profile";
import { Task } from "./Task";

export function Game() {
  const game = useGame();
  const stage = useStage();
  const player = usePlayer();
  const { playerCount } = game.get("treatment");

  return (
    <div className="h-full w-full flex">
      <div className="h-full w-full flex flex-col">
        <Profile />
        <div className="h-full flex items-center justify-center">
          <Task />
        </div>
      </div>

      {player.get("group") == "red" && (
        <div className="h-full w-128 border-l flex justify-center items-center">
          <Chat player={player} scope={stage} attribute="red_chat" />
        </div>
      )}

      {player.get("group") == "blue" && (
        <div className="h-full w-128 border-l flex justify-center items-center">
          <Chat player={player} scope={stage} attribute="blue_chat" />
        </div>
      )}
    </div>
  );
}
