#!/bin/sh
# Downloads an MP4 video from MinIO and streams it via FFmpeg to MediaMTX as an RTSP
# source. Used when no real camera is available: provides an in-cluster simulated
# video feed for the PPE compliance monitor (e.g. rtsp://video-stream:8554/live).
set -e

export MC_CONFIG_DIR=/tmp/.mc

MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://minio:9000}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"
MINIO_VIDEO_BUCKET="${MINIO_VIDEO_BUCKET:-data}"
MINIO_VIDEO_KEY="${MINIO_VIDEO_KEY:-combined-video-no-gap-rooftop.mp4}"
MEDIAMTX_HOST="${MEDIAMTX_HOST:-video-stream}"
MEDIAMTX_PORT="${MEDIAMTX_PORT:-8554}"
VIDEO_PATH="/tmp/video.mp4"

echo "=== Video Stream Publisher ==="
echo "Waiting for MinIO..."
until mc alias set myminio "${MINIO_ENDPOINT}" "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}" 2>/dev/null; do
	echo "MinIO not ready, retrying in 2 seconds..."
	sleep 2
done
echo "MinIO connected"

echo "Downloading video from MinIO..."
mc cp "myminio/${MINIO_VIDEO_BUCKET}/${MINIO_VIDEO_KEY}" "${VIDEO_PATH}"
echo "Video downloaded"

echo "Waiting for MediaMTX to be ready..."
until nc -z "${MEDIAMTX_HOST}" "${MEDIAMTX_PORT}" 2>/dev/null; do
	echo "MediaMTX not ready, retrying in 2 seconds..."
	sleep 2
done
echo "MediaMTX ready"

echo "Starting FFmpeg stream (loop)..."
exec ffmpeg -re -stream_loop -1 -i "${VIDEO_PATH}" -c copy -f rtsp "rtsp://${MEDIAMTX_HOST}:${MEDIAMTX_PORT}/live"
