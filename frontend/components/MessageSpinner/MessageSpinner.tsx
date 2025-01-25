import React from "react";
import "./MessageSpinner.css";

const MessageSpinner: React.FC = () => {
  return (
    <div className="message-spinner">
      <div className="spinner">
        <div className="bounce1"></div>
        <div className="bounce2"></div>
        <div className="bounce3"></div>
      </div>
    </div>
  );
};

export default MessageSpinner; 