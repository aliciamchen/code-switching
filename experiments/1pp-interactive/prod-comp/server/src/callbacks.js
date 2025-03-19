import { ClassicListenersCollector } from "@empirica/core/admin/classic";
export const Empirica = new ClassicListenersCollector();
import _ from "lodash";
import {
  tangram_sets,
  names,
  name_colors,
  conditions,
  avatar_names,
} from "./constants";

Empirica.onGameStart(({ game }) => {
  console.log("Game started");
  // Assign tangram set
  const tangram_set = _.random(0, 2);
  const context = tangram_sets[tangram_set];
  game.set("tangram_set", tangram_set);
  game.set("context", context);

  game.players.forEach((player, i) => {
    player.set("name", names[i]);
    player.set("tangram_set", tangram_set);
    player.set("context", context);
    player.set("score", 0);
    player.set("bonus", 0);
  });

  // Randomly assign 4 of the players to the red group and the other 4 to the blue group
  red_players = _.sampleSize(game.players, game.players.length / 2);
  blue_players = _.difference(game.players, red_players);
  red_players.forEach((player, i) => {
    player.set("group", "red");
    player.set("player_index", i);
    player.set("avatar_name", avatar_names["red"][i]);
    player.set("avatar", `/${avatar_names["red"][i]}.png`);
    player.set("name_color", name_colors["red"][i]);
    const shuffled_tangrams = _.shuffle(context);
    player.set("shuffled_tangrams", shuffled_tangrams);
    player.set(
      "tangramURLs",
      shuffled_tangrams.map((tangram) => `/tangram_${tangram}.png`)
    );
  });
  blue_players.forEach((player, i) => {
    player.set("group", "blue");
    player.set("player_index", i);
    player.set("avatar_name", avatar_names["blue"][i]);
    player.set("avatar", `/${avatar_names["blue"][i]}.png`);
    player.set("name_color", name_colors["blue"][i]);
    const shuffled_tangrams = _.shuffle(context);
    player.set("shuffled_tangrams", shuffled_tangrams);
    player.set(
      "tangramURLs",
      shuffled_tangrams.map((tangram) => `/tangram_${tangram}.png`)
    );
  });

  // PHASE 1: REFERENCE GAME
  // Players play a reference game within their group
  // The speaker has to refer to each tangram in the context before the speaker role moves to the next player for the next block of trials.
  // There are 8 blocks total (so each player will be in the speaker role twice, and each tangram will appear as the target exactly 8 times)
  // _.times(2, (i) => {
  //   // loop through each speaker twice
  //   _.times(4, (player_index) => {
  //     // loop through each player in the group
  //     const shuffled_context = _.shuffle(context);
  //     _.times(shuffled_context.length, (target_num) => {
  //       // in each round (repetition block), loop through each tangram in the context
  //       const round = game.addRound({
  //         name: `Reference Game`,
  //         phase: "refgame",
  //         speaker: player_index,
  //         target_order: shuffled_context,
  //         target: shuffled_context[target_num],
  //         target_num: target_num,
  //         rep_num: i * 4 + player_index,
  //       });
  //       round.addStage({
  //         name: "Selection",
  //         duration: 6000,
  //       });
  //       round.addStage({
  //         name: "Feedback",
  //         duration: 6000,
  //       });
  //     });
  //   });
  // });

  // game.addRound({
  //   name: "End of Phase 1",
  //   duration: 30,
  // });

  // PHASE 2: SPEAKER PRODUCTION

  const phase_2 = game.addRound({
    name: "Phase 2", // change later
    phase: "speaker_prod",
  });
  const tangram_combos = _.flatten(
    context.map((tangram) =>
      conditions.map((condition) => [tangram, condition])
    )
  ); // all combos of tangrams and conditions
  // for each player, add the randomized order of tangram-condition pairs
  game.players.forEach((player) => {
    player.set("phase_2_trial_order", _.shuffle(tangram_combos));
    // console.log(player.get("phase_2_trial_order"));
  });
  _.times(tangram_combos.length, (i) => {
    phase_2.addStage({
      name: "Production",
      duration: 6000,
      trial_num: i,
    }); // each player sees a different order of tangram-condition pairs, so the trial number is used to index into the player's order in Stage.jsx
  });

  // PHASE 3: LISTENER INTERPRETATION

  const phase_3 = game.addRound({
    name: "Phase 3", // change later
    phase: "comprehension",
  });
  // TODO: write function to take the phase 2 utterances, and assign them to players in phase 3
  // the phase 3 utterances should be collected in the player variable?
});

Empirica.onRoundStart(({ round }) => {
  // On refgame round starts, set player roles
  if (round.get("phase") == "refgame") {
    const players = round.currentGame.players;
    const red_players = players.filter(
      (player) => player.get("group") == "red"
    );
    const blue_players = players.filter(
      (player) => player.get("group") == "blue"
    );
    const speaker = round.get("speaker");
    red_players.forEach((player, i) => {
      // player.set("clicked", ""); // refresh clicked state
      player.round.set("role", i == speaker ? "speaker" : "listener");
    });
    blue_players.forEach((player, i) => {
      // player.set("clicked", "");
      player.round.set("role", i == speaker ? "speaker" : "listener");
    });
  }
});

Empirica.onStageStart(({ stage }) => {
  if (stage.get("name") === "Selection") {
    stage.set("red_chat", []);
    stage.set("blue_chat", []);
  }

  // Each participant sees the trials in a different order, so we want to add that information to player stages
  if (stage.get("name") === "Production") {
    const game = stage.currentGame;
    const players = game.players;

    players.forEach((player) => {
      const trial_num = stage.get("trial_num");
      const trial = player.get("phase_2_trial_order")[trial_num];

      player.stage.set("target", trial[0]);
      player.stage.set("condition", trial[1]);
    });
  }
});

Empirica.onStageEnded(({ stage }) => {
  if (stage.get("name") === "Selection") {
    // Calculate score for the current stage
    // Listeners get 1 point when they correctly identify the target
    // The speaker gets the average of the listeners' scores, in that group.

    const game = stage.currentGame;
    const players = game.players;
    const round = stage.round;

    const target = round.get("target");
    const red_players = players.filter(
      (player) => player.get("group") == "red"
    );
    const blue_players = players.filter(
      (player) => player.get("group") == "blue"
    );

    const red_listeners = red_players.filter(
      (player) => player.round.get("role") == "listener"
    );
    const blue_listeners = blue_players.filter(
      (player) => player.round.get("role") == "listener"
    );

    const red_speaker = red_players.find(
      (player) => player.round.get("role") == "speaker"
    );
    const blue_speaker = blue_players.find(
      (player) => player.round.get("role") == "speaker"
    );

    const red_correct = red_listeners.filter(
      (player) => player.round.get("clicked") == target
    );
    const blue_correct = blue_listeners.filter(
      (player) => player.round.get("clicked") == target
    );

    red_correct.forEach((player) => {
      player.set("score", player.get("score") + 3);
    });
    blue_correct.forEach((player) => {
      player.set("score", player.get("score") + 3);
    });

    const red_avg_score = red_listeners.length
      ? (red_correct.length * 3) / red_listeners.length
      : 0;
    const blue_avg_score = blue_listeners.length
      ? (blue_correct.length * 3) / blue_listeners.length
      : 0;

    red_speaker.set("score", red_speaker.get("score") + red_avg_score);
    red_speaker.round.set("round_score", red_avg_score);
    blue_speaker.set("score", blue_speaker.get("score") + blue_avg_score);
    blue_speaker.round.set("round_score", blue_avg_score);
  }

  // Add Phase 2 speaker utterances to player's round data (for collecting later)
  if (stage.get("name") === "Production") {
    const game = stage.currentGame;
    const players = game.players;

    players.forEach((player) => {
      const utterance = player.stage.get("utterance");

      if (!player.round.get("utterances")) {
        player.round.set("utterances", {});
      }

      const player_utterances = player.round.get("utterances");
      console.log(player_utterances);

      const condition = player.stage.get("condition");
      const tangram = player.stage.get("target");

      if (utterance) {
        // create condition, tangram keys if they don't exist
        if (!player_utterances[condition]) {
          player_utterances[condition] = {};
        }
        if (!player_utterances[condition][tangram]) {
          player_utterances[condition][tangram] = {};
        }

        // add utterance to player's utterances
        player_utterances[condition][tangram] = utterance;
        player.round.set("utterances", player_utterances);
      }
    });
  }
});

Empirica.onRoundEnded(({ round }) => {
  // TODO: If we are at the end of phase 2, collect the utterances based on condition and group
});

Empirica.onGameEnded(({ game }) => {});
