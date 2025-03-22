import React from "react";
import { Tangram } from "../components/Tangram.jsx";

// Participants are asked to
// generate the best labels for inducing a particular audience (a member of their ‘own’ group or a member of the
//     ‘other’ group) to make one of two inferences (the ‘referential’ goal of choosing which tangram is being referred
//     to, or the ‘social’ goal of being identified as a member of the audience group).
export function Production() {
  const { round, stage, game, player, players } = props;
  const [description, setDescription] = React.useState("");

  const target = player.stage.get("target");
  const condition = player.stage.get("condition");

  let prompt;
  if (condition == "refer own") {
    prompt = (
      <>
        Please write a description to help a member of{" "}
        <span style={{ fontWeight: "bold" }}>your own</span> group{" "}
        <span style={{ fontWeight: "bold" }}>pick the correct picture</span>.
      </>
    );
  } else if (condition == "refer other") {
    prompt = (
      <>
        Please write a description to help a member of{" "}
        <span style={{ fontWeight: "bold" }}>another</span> group{" "}
        <span style={{ fontWeight: "bold" }}>pick the correct picture</span>.
      </>
    );
  } else if (condition == "social own") {
    prompt = (
      <>
        Please write a description of the target picture, to help a member of{" "}
        <span style={{ fontWeight: "bold" }}>your own</span> group{" "}
        <span style={{ fontWeight: "bold" }}>
          identify you as a member of their group
        </span>
        .
      </>
    );
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

  const handleSubmit = (e) => {
    e.preventDefault();
    player.stage.set("utterance", description);
    player.stage.set("submit", true);
  };

  if (player.stage.get("submit")) {
    return (
      <div className="task">
        <div className="board">
          <div className="prompt-container">
            <p className="instruction-prompt">Waiting for other players...</p>
          </div>
        </div>
      </div>
    );
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

      <div className="description-container">
        <form onSubmit={handleSubmit}>
          <textarea
            className="description-input"
            placeholder="Write your description here..."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={4}
            required
          />
          <div className="submit-container">
            <button
              type="submit"
              className="submit-button"
              disabled={!description.trim()}
            >
              Submit Description
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
