import { useStageTimer } from "@empirica/core/player/classic/react";
import React from "react";
import _ from "lodash";
import { useGame } from "@empirica/core/player/classic/react";

//TODO: click didnt register because speaker didnt say anythingth

export function Tangram(props) {
  const {
    tangram,
    tangram_num,
    stage,
    player,
    players,
    round,
    game,
    target,
    ...rest
  } = props;

  // Red group and blue groups play games separately
  const playerGroup = player.get("group");
  const playersInGroup = players.filter((p) => p.get("group") == playerGroup);
  const playerGroupSpeaker = playersInGroup.filter(
    (p) => p.round.get("role") == "speaker"
  )[0];

  const handleClick = (e) => {
    console.log("click2");

    const playerGroupChat = stage.get(`${playerGroup}_chat`);
    const speakerMsgs = _.filter(playerGroupChat, (msg) => {
      return msg.sender.id == playerGroupSpeaker.id;
    });

    // only register click for listener and only after the speaker has sent a message
    if (
      (stage.get("name") == "Selection") &
      (speakerMsgs.length > 0) &
      !player.stage.get("clicked") &
      (player.round.get("role") == "listener")
    ) {
      player.stage.set("clicked", tangram);
      player.round.set("lastClicked", tangram);
      setTimeout(() => player.stage.set("submit", true), 3000); // FIX?? this does not submit
    }
    // end stage if all listeners have clicked
    const listeners = playersInGroup.filter(
      (p) => p.round.get("role") == "listener"
    );
    const allClicked = _.every(listeners, (p) => p.stage.get("clicked"));
    console.log(player.stage.get("clicked"));
    console.log(allClicked);
    if (allClicked) {
        console.log(playersInGroup)
      // end stage
      playersInGroup.forEach((p) => {
        console.log("xxx")
        p.stage.set("submit", true);
      });

      // TODO: when all in-group listeners have clicked, display something about waiting for everyone to respond... figure out how to do this in a way that makes sense
    }
  };

  const row = 1 + Math.floor(tangram_num / 3);
  const column = 1 + (tangram_num % 3);
  const mystyle = {
    background: "url(tangram_" + tangram + ".png)",
    backgroundSize: "cover",
    width: "20vh",
    height: "20vh",
    gridRow: row,
    gridColumn: column,
  };

  // Highlight target object for speaker
  if ((target == tangram) & (player.round.get("role") == "speaker")) {
    _.extend(mystyle, {
      outline: "10px solid #000",
      zIndex: "9",
    });
  }

  // Show listeners what they've clicked
  if (
    (stage.get("name") == "Selection") &
    (tangram == player.stage.get("clicked"))
  ) {
    _.extend(mystyle, {
      outline: `10px solid #A9A9A9`,
      zIndex: "9",
    });
  }

  // Feedback
  // TODO: change this to be based on round.... not stage. Right now it doesn't work because the stage is over by the time the feedback is displayed
  let feedback = [];
  if (stage.get("name") == "Feedback") {
    playersInGroup.forEach((p) => {
      if (p.round.get("lastClicked") == tangram) {
        feedback.push(<img src={player.get("avatar")} key="player" />);
      }
    });
  }
  if (
    (stage.get("name") == "Feedback") &
    _.some(playersInGroup, (p) => p.round.get("lastClicked") == tangram)
  ) {
    const color = tangram == target ? "green" : "red";
    _.extend(mystyle, {
      outline: `10px solid ${color}`,
      zIndex: "9",
    });
  }

  return (
    <div onClick={handleClick} style={mystyle}>
      <div className="feedback"> {feedback}</div>
    </div>
  );
}
