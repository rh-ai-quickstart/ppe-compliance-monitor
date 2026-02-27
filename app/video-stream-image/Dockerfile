# Video stream publisher: downloads MP4 from MinIO and streams via RTSP to MediaMTX
FROM docker.io/minio/mc:latest

# Install ffmpeg and netcat (mc image is based on Alpine)
RUN apk add --no-cache ffmpeg netcat-openbsd

WORKDIR /stream

COPY stream-from-minio.sh /stream/stream-from-minio.sh
RUN chmod +x /stream/stream-from-minio.sh

ENTRYPOINT ["/stream/stream-from-minio.sh"]
