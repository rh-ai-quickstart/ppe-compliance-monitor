import React from 'react';
import './LogoBar.css';

const LogoBar = () => {
  return (
    <div className="logo-bar">
      <img src="/redhat-logo.png" alt="Red Hat Logo" className="logo redhat-logo" />
      <img src="/intel-logo.jpg" alt="Intel Logo" className="logo intel-logo" />
    </div>
  );
};

export default LogoBar;