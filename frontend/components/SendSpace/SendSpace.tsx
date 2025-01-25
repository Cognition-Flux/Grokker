"use client";
import React, { useEffect, useRef, useState } from "react";
import { Send } from "lucide-react";
import "./SendSpace.css";
import { Message } from "@/lib/interfaces";
import { StreamData } from "@/lib/chat_api";
import { LoadingAnimated } from "../LoadingAnimated";

export type SendSpaceProps = {
  addMessage: (icon: string, text: string) => void;
  setStreamMessages: (x: Message) => void;
  setLoading: (x: boolean) => void;
  readIdSession: () => string;
  loading: boolean;
  selectedOffices: string[];
};

const SendSpace: React.FC<SendSpaceProps> = ({
  addMessage,
  setStreamMessages,
  setLoading,
  loading,
  readIdSession,
  selectedOffices,
}) => {
  const [inputText, setInputText] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const handleInputChange = (e: any) => {
    setInputText(e.target.value);
  };
  const saveInputToMessage = () => {
    addMessage("human", inputText);
  };
  const queryLLM = () => {
    saveInputToMessage();
    StreamData(
      inputText,
      addMessage,
      setLoading,
      setStreamMessages,
      readIdSession,
      selectedOffices
    );
  };

  const handleKeyUp = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey && inputText) {
      e.preventDefault();
      queryLLM();
    }
  };

  useEffect(() => {
    if (loading) setInputText("Cargando...");
    else {
      setInputText("");
      inputRef.current?.focus();
    }
  }, [loading]);

  return (
    <div className="chat-input">
      {loading ? (
        <LoadingAnimated setLoading={setLoading} />
      ) : (
        <div className="sendspace">
          <input
            disabled={loading}
            value={inputText}
            onChange={handleInputChange}
            onKeyUp={handleKeyUp}
            placeholder="Escribe un mensaje..."
            className="input"
            ref={inputRef}
          />
          <button
            disabled={loading || !inputText}
            onClick={queryLLM}
            className="button"
          >
            <Send />
          </button>
        </div>
      )}
      <style jsx>{`
        .chat-input {
          align-self: end;
          grid-area: chat-input;
        }
      `}</style>
    </div>
  );
};

export default SendSpace;
