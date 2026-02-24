import React from 'react';
import LiveFeedLabel from './LiveFeedLabel';
import './VideoFeed.css'; // Assuming you have some styles for your video feed
import { API_URL } from '../config';
const VideoFeed = () => {
  return (
    <div className="video-feed-container">
      <LiveFeedLabel />
      <img src={`${API_URL}/video_feed`} alt="Video Feed" />
    </div>
  );
};

export default VideoFeed;
