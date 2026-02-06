import React from 'react';
import './VideoPlayer.css';
import LiveFeedLabel from './LiveFeedLabel';
import { API_URL } from '../config';

// Use `${API_URL}/latest_info` for your API calls
const VideoPlayer = () => {
  return (
    <div className="video-feed-container">
      <LiveFeedLabel />
      <img src={`${API_URL}/video_feed`} alt="Video Feed" />
    </div>
  );
};

export default VideoPlayer;