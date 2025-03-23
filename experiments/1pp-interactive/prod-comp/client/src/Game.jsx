import {
  Chat,
  usePlayer,
  usePlayers,
  useRound,
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
  const players = usePlayers();
  const round = useRound();


  // right now this is repeating code - fix later
  const playerGroup = player.get("group");
  const playersInGroup = players.filter(
    (p) => p.get("group") === playerGroup
  );
  const allGroupResponded = playersInGroup.every(
    (p) => p.round.get("role") === "speaker" || p.round.get("clicked")
  );

  return (
    <div className="h-full w-full flex">
      <div className="h-full w-full flex flex-col">
        <Profile />
        <div className="h-full flex items-center justify-center">
          <Task round={round} stage={stage} game={game} player={player} players={players}/>
        </div>
      </div>

      {player.get("group") == "red" && stage.get("name") == "Selection" && !allGroupResponded && (
        <div className="h-full w-128 border-l flex justify-center items-center">
          <Chat player={player} scope={stage} attribute="red_chat" />
        </div>
      )}

      {player.get("group") == "blue" && stage.get("name") == "Selection" && !allGroupResponded && (
        <div className="h-full w-128 border-l flex justify-center items-center">
          <Chat player={player} scope={stage} attribute="blue_chat" />
        </div>
      )}
    </div>
  );
}
