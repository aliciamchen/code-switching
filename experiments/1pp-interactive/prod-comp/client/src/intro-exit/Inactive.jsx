import React from "react";
import { usePlayer } from "@empirica/core/player/classic/react";
import { Alert } from "../components/Alert";
import { Button } from "../components/Button";

export function Inactive({ next }) {

  const player = usePlayer();

  function handleSubmit(event) {
    event.preventDefault();
    player.set("exitSurvey", {
      "gamefailed": "nogame"
    })
    next();
  }

  return (
    <div className="py-8 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
      <Alert title="No Game Available">
        <p>
          Unfortunately, one of the players is inactive, so we had to end the game. You will still be compensated for your time. 
        </p>
      </Alert>
      <Alert title="Payment">
        <p>
        Please submit the following code to receive a partial payment:{" "}
          <strong>C17Y1U5H</strong>
        </p>
        <p className="pt-1">
          Thank you for your time and willingness to participate in our study.
        </p>
      </Alert>

      <form onSubmit={handleSubmit}>
        <div className="mt-8">
          <Button type="submit">Submit</Button>
        </div>
      </form>
    </div>
  );
}