import React, { useState } from 'react';
import axios from 'axios';

function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const handleGetAnswer = async () => {
    if (input.trim() === '') return;
  
    const userMessage = { text: input, isUser: true };
    setMessages([...messages, userMessage]);
    setInput('');
  
    try {
      const response = await axios.get(`http://localhost:8005/answer?query=${encodeURIComponent(input)}`);
      const botResponse = { text: response.data.answer, isUser: false };
      setMessages(prevMessages => [...prevMessages, botResponse]);
    } catch (error) {
      console.error('Error getting answer:', error);
      const errorMessage = { text: "Sorry, I couldn't get an answer.", isUser: false };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    }
  };
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (input.trim() === '') return;

    const userMessage = { text: input, isUser: true };
    setMessages([...messages, userMessage]);
    setInput('');

    try {
      const response = await axios.post('http://localhost:8005/query', { query: input });
      const botResponse = { text: response.data.response, isUser: false };
      setMessages(prevMessages => [...prevMessages, botResponse]);
    } catch (error) {
      console.error('Error sending query:', error);
      const errorMessage = { text: "Sorry, I couldn't process your request.", isUser: false };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    }
  };

  return (
    <div className="chatbot">
      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.isUser ? 'user' : 'bot'}`}>
            {message.text}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
        />
        <button type="submit">Send</button>
        <button type="button" onClick={handleGetAnswer}>Get Answer</button>
      </form>
    </div>
  );
}

export default Chatbot;