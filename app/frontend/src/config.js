const runtimeConfig = window.__ENV__ || {};

const normalizeApiUrl = (value) => {
  if (!value) return '';
  if (value.startsWith('/')) {
    return value;
  }
  if (value.startsWith('http://') || value.startsWith('https://')) {
    return value;
  }
  try {
    return new URL(`https://${value}`).toString().replace(/\/$/, '');
  } catch (error) {
    return '';
  }
};

const configuredApiUrl = normalizeApiUrl(
  runtimeConfig.API_URL || process.env.REACT_APP_API_URL
);

export const API_URL =
  configuredApiUrl ||
  `${window.location.origin}/api` ||
  'http://localhost:8888';
