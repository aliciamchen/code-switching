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
    (p) => p.get("role") == "speaker"
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
      !player.get("clicked") &
      (player.get("role") == "listener")
    ) {
      // for each player in the group
      playersInGroup.forEach((p) => {
        p.set("clicked", tangram);
        setTimeout(() => p.stage.set("submit", true), 3000);
      });
    }
  };

  const row = 1 + Math.floor(tangram_num / 3);
  const column = 1 + (tangram_num % 3);
  const mystyle = {
    background: "url(tangram_" + tangram + ".png)",
    backgroundSize: "cover",
    width: "25vh",
    height: "25vh",
    gridRow: row,
    gridColumn: column,
  };

  // Highlight target object for speaker
  if ((target == tangram) & (player.get("role") == "speaker")) {
    _.extend(mystyle, {
      outline: "10px solid #000",
      zIndex: "9",
    });
  }

  // Show listeners what they've clicked
  if ((stage.get("name") == "Selection") & (tangram == player.get("clicked"))) {
    _.extend(mystyle, {
      outline: `10px solid #A9A9A9`,
      zIndex: "9",
    });
  }

  // Feedback
  // TODO: right now, listeners can't select different tangrams
  let feedback = [];
  if (stage.get("name") == "Feedback") {
    playersInGroup.forEach((p) => {
      if (p.get("clicked") == tangram) {
        feedback.push(<img src={player.get("avatar")} key="player" />);
      }
    });
  }
  if (
    (stage.get("name") == "feedback") &
    _.some(playersInGroup, (p) => p.get("clicked") == tangram)
  ) {
    const color = tangram == target ? "green" : "red";
    _.extend(mystyle, {
      outline: `10px solid ${color}`,
      zIndex: "9",
    });
  }

  // Highlight target object for speaker at selection stage
  // Show it to both players at feedback stage if 'showNegativeFeedback' enabled.
  //   if (tangram == target) {
  //     if (player.get("role") == "speaker" || player.get("clicked")) {
  //       _.extend(mystyle, {
  //         outline: "10px solid #000",
  //         zIndex: "9",
  //       });
  //     }
  //     if (player.get("role") == "speaker" && player.get("clicked")) {
  //       _.extend(mystyle, {
  //         outline: "10px solid red",
  //         zIndex: "9",
  //       });
  //     }
  //   }

  return (
    <div onClick={handleClick} style={mystyle}>
      <div className="feedback"> {feedback}</div>
    </div>
  );
}
