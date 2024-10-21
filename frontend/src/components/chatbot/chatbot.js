import React, { useState } from 'react';
import './chatbot.css';

function Chatbot() {
  const [userMessage, setUserMessage] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [recommendedSongs, setRecommendedSongs] = useState([]);

  const handleSendMessage = async () => {
    if (!userMessage.trim()) return;

    // Add user message to chat
    setChatMessages([...chatMessages, { sender: 'user', text: userMessage }]);
    
    try {
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
      });

      const data = await response.json();

      // Add chatbot response to chat
      setChatMessages((prevMessages) => [
        ...prevMessages,
        { sender: 'bot', text: data.chatbot_response },
      ]);

      // Update the recommended songs
      setRecommendedSongs(data.recommended_songs || []);

    } catch (error) {
      console.error('Error:', error);
    }

    setUserMessage('');
  };

  return (
    <div className="chatbot-container">
      <div className="chat-column">
        {chatMessages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            {msg.text}
          </div>
        ))}
        <div>
          <input
            type="text"
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
            placeholder="Type your message..."
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          />
          <button onClick={handleSendMessage}>Send</button>
        </div>
      </div>
      <div className="songs-column">
        <h2>Recommended Songs</h2>
        {recommendedSongs.length > 0 ? (
          <ul>
            {recommendedSongs.map((song, index) => (
              <li key={index}>
                {song}
              </li>
            ))}
          </ul>
        ) : (
          <p>No recommendations available.</p>
        )}
      </div>
    </div>
  );
}

export default Chatbot;
