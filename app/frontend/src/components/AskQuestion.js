import React, { useState } from 'react';
import axios from 'axios';
import { API_URL } from '../config';

const AskQuestion = () => {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post(`${API_URL}/ask_question`, { question });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error('Error asking question:', error);
    }
  };

  return (
    <div className="ask-question">
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question..."
        />
        <button type="submit">Ask</button>
      </form>
      {answer && <p className="answer">{answer}</p>}
    </div>
  );
};

export default AskQuestion;
