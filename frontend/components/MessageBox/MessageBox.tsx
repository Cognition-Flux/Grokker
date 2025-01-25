"use client";
import React from "react";
import "./MessageBox.css";
import ReactMarkdown from "react-markdown";
import gfm from "remark-gfm";
import { CircleUserRound, Network } from "lucide-react";
import FeedbackComponent from "./FeedbackComponent";

export type MessageBoxProps = {
  icon: string;
  text: string;
  run_id: string | undefined;
  feedback: number;
  feedback_text: string[] | undefined;
  update_feedback: (
    run_id: string,
    feedback: number,
    feedback_text: string[]
  ) => void;
  showFeedback?: boolean;
};

const MessageBox: React.FC<MessageBoxProps> = ({
  icon,
  text,
  run_id,
  feedback = -1,
  feedback_text = undefined,
  update_feedback,
  showFeedback = true,
}) => {
  const role = icon === "human" ? "user" : "assistant";
  
  return (
    <div className="messagebox" data-role={role}>
      <div className="icon">
        {icon == "human" ? (
          <CircleUserRound className="flex-none h-7 w-7 mt-1" />
        ) : (
          <Network className="flex-none h-7 w-7 mt-1" />
        )}
      </div>
      <div className="message rounded-lg transition-all duration-300 ease-in-out">
        <ReactMarkdown
          remarkPlugins={[gfm]}
          components={{
            h1: ({ node, ...props }) => (
              <h1 className="font-bold text-3xl" {...props} />
            ),
            h2: ({ node, ...props }) => (
              <h2 className="font-bold text-2xl" {...props} />
            ),
            h3: ({ node, ...props }) => (
              <h3 className="font-bold text-xl" {...props} />
            ),
            p: ({ node, ...props }) => (
              <p
                className="text-wrap leading-relaxed hyphens-auto mb-4"
                {...props}
              />
            ),
            // TODO: Implement a copy paste for the tables
            table: ({ node, ...props }) => (
              <table className="my-2 hover:border-gray-100" {...props} />
            ),

            th: ({ node, ...props }) => (
              <th className="border-2 py-1 px-2" {...props} />
            ),
            td: ({ node, ...props }) => (
              <td className="border-2 py-1 px-2" {...props} />
            ),
            ul: ({ node, ...props }) => (
              <ul
                className="list-disc list-inside marker:text-gray-400"
                {...props}
              />
            ),
            li: ({ node, ...props }) => (
              <li
                className="list-decimal list-inside marker:text-gray-400"
                {...props}
              />
            ),
          }}
        >
          {text}
        </ReactMarkdown>
        {showFeedback && run_id && (
          <FeedbackComponent
            run_id={run_id}
            feedback={feedback}
            feedback_text={feedback_text}
            update_feedback={update_feedback}
          />
        )}
      </div>
    </div>
  );
};

export default MessageBox;
