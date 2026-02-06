import React, { useState } from 'react';
import VideoPlayer from './components/VideoPlayer';
import PPEDescription from './components/PPEDescription';
import ChatBot from './components/ChatBot';
import LogoBar from './components/LogoBar';
import './App.css';
import './App.custom.css';
import architectureDiagram from './itap-demo.png'; // Make sure this path is correct

function App() {
  const [showDiagram, setShowDiagram] = useState(false);

  const toggleDiagram = () => {
    setShowDiagram(!showDiagram);
  };

  return (
    <div className="App">
      <button className="diagram-toggle" onClick={toggleDiagram}>
        Architecture Diagram
      </button>
      
      {showDiagram && (
        <div className="diagram-overlay">
          <img src={architectureDiagram} alt="Architecture Diagram" />
        </div>
      )}

      <LogoBar />
      <h1 className="main-title">
        Multi Modal and Multi Model Safety Monitoring System
        <span className="company-names">
          by <span className="intel">Intel</span> and <span className="redhat">Red Hat</span>
        </span>
      </h1>
      <div className="content-wrapper">
        <div className="left-content">
          <VideoPlayer />
          <PPEDescription />
        </div>
        <div className="right-content">
          <ChatBot />
        </div>
      </div>
    </div>
  );
}

export default App;
