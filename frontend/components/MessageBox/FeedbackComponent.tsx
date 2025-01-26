"use client";
import React from "react";
import "./MessageBox.css";
import { Toggle } from "../ui/toggle";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import AppSettings from "@/lib/AppSettings.json";
import { Button } from "../ui/button";

export type MessageBoxProps = {
  run_id: string;
  feedback: number;
  feedback_text: string[] | undefined;
  update_feedback: (
    run_id: string,
    feedback: number,
    feedback_text: string[]
  ) => void;
};

const MessageBox: React.FC<MessageBoxProps> = ({
  run_id,
  feedback,
  feedback_text,
  update_feedback,
}) => {
  /* TODO: Implement a multi select feedback
   * - "Informacion incorrecta"
   * - "Contexto incorrecto"
   * - "Respuesta incompleta"
   * - "Respuesta muy extensa"
   */

  const handle_button = (new_feedback: number) => {
    // If the feedback is the same, reset the feedback
    // This is explicit for clarity
    if (feedback === new_feedback) {
      console.log("Resetting feedback");
      feedback = -1;
    } else {
      console.log("Setting feedback: ", new_feedback);
      feedback = new_feedback;
    }

    // Set the state here
    update_feedback(run_id, feedback, []);

    // Send feedback to API
    fetch(AppSettings.BaseLlmUrl + "/feedback", {
      method: "POST",
      headers: {
        accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        run_id: run_id,
        key: "human-feedback-thumb",
        score: new_feedback === -1 ? feedback : new_feedback,
        kwargs: {
          comment: feedback_text?.join(", "), // TODO: Implement a comment
        },
      }),
    });
  };

  return (
    <div className="px-2 border border-[#FF8C00] flex gap-2 h-6">
      <span className="text-gray-500 mt-1 text-sm">Feedback</span>
      <button onClick={() => handle_button(1)}>
        <ThumbsUp
          className={
            feedback == 1
              ? "h-4 w-4 stroke-green-500 hover:stroke-green-400"
              : "h-4 w-4 stroke-gray-500 hover:stroke-gray-400"
          }
        />
      </button>
      <button onClick={() => handle_button(0)}>
        <ThumbsDown
          className={
            feedback === -1
              ? "h-4 w-4 stroke-gray-500 hover:stroke-gray-400"
              : feedback === 0
              ? "h-4 w-4 stroke-red-500 hover:stroke-red-400"
              : "h-4 w-4 stroke-gray-500 hover:stroke-gray-400"
          }
        />
      </button>
    </div>
  );
};

/*       
    {feedback && (<Toggle variant="outline" className="h-6">
        Informacion incorrecta
      </Toggle>
      <Toggle variant="outline" className="h-6">
        Contexto incorrecto
      </Toggle>
      <Toggle variant="outline" className="h-6">
        Respuesta incompleta
      </Toggle>
      <Toggle variant="outline" className="h-6">
        Respuesta muy extensa
      </Toggle>)}
 */

export default MessageBox;
