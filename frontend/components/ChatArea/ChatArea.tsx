"use client";
import React, { useEffect, useRef } from "react";
import { Message } from "@/lib/interfaces";
import MessageBox from "../MessageBox/MessageBox";
import "./ChatArea.css";
import tagentig_logo from "@/lib/assets/GROKKER.svg";
import Image from "next/image";

export type ChatAreaProps = {
  messages: Message[];
  stremMessage: Message;
  loading: boolean;
  update_feedback: (
    run_id: string,
    feedback: number,
    feedback_text: string[]
  ) => void;
};

const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
  stremMessage,
  loading,
  update_feedback,
}) => {
  //referencia al contenedor de mensajes
  const listRef = useRef<HTMLDivElement>(null);
  //scroleamos al ultimo elemento con scrollIntoView
  useEffect(() => {
    // Desplaza el contenedor al final cuando se agregue un mensaje nuevo
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, stremMessage]);
  return (
    <div className="chat-area scroll-smooth" ref={listRef}>
      {messages.length <= 0 && (
        <div className="flex items-center justify-center h-full">
          <Image src={tagentig_logo} height={300} alt="TotalPack" />
        </div>
      )}

    {messages.map((aux, index) => (
        <MessageBox
          key={index}
          icon={aux.icon}
          text={aux.text}
          run_id={aux.run_id}
          feedback={aux.feedback}
          feedback_text={aux.feedback_text}
          update_feedback={update_feedback}
          showFeedback={false}
        />
      ))}
      {loading && (
        <MessageBox
          icon={stremMessage.icon}
          text={stremMessage.text}
          run_id={undefined}
          feedback={-1}
          feedback_text={undefined}
          update_feedback={update_feedback}
          showFeedback={false}
        />
      )}
      <style jsx>{`
        .chat-area {
          grid-area: chat-area;
          overflow-y: auto;
        }
      `}</style>
    </div>
  );
};

export default ChatArea;
