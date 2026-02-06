import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './PPEDescription.css';
import { API_URL } from '../config';

const PPEDescription = () => {
  const [description, setDescription] = useState('Initializing...');
  const [summaries, setSummaries] = useState([]);

  useEffect(() => {
    const fetchLatestInfo = async () => {
      try {
        const response = await axios.get(`${API_URL}/latest_info`);
        setDescription(response.data.description);
        setSummaries((prevSummaries) => [
          { text: response.data.summary, isCurrent: true },
          ...prevSummaries.slice(0, 2).map(summary => ({ ...summary, isCurrent: false })),
        ]);
      } catch (error) {
        console.error('Error fetching latest info:', error);
      }
    };

    const intervalId = setInterval(fetchLatestInfo, 5000);
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="ppe-description">
      <div className="description-section">
        <h3>Latest PPE Detection</h3>
        <p className="detection-info">{description}</p>
      </div>
      <div className="summary-section">
        <h3>Safety Trends</h3>
        <div className="summary-feed">
          {summaries.map((summary, index) => (
            <div
              key={index}
              className={`safety-trends ${summary.isCurrent ? 'current-summary' : ''}`}
            >
              <pre>{summary.text}</pre>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default PPEDescription;