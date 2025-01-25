import React, { useState } from 'react';

const MessageCloud = () => {
  const [isLoading, setIsLoading] = useState(false);

  return (
    <div className="message-container">
      {messages.map((message, index) => (
        <div key={index} className={`message ${message.role}`}>
          {message.content}
        </div>
      ))}
      
      {isLoading && (
        <div className="message assistant loading">
          <div className="spinner">
            <div className="bounce1"></div>
            <div className="bounce2"></div>
            <div className="bounce3"></div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MessageCloud; 