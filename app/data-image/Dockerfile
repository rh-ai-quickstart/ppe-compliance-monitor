# Dockerfile for ppe-compliance-monitor-data
# Uploads model and video files to MinIO (idempotent - skips if already exists)

# Stage 1: Extract compressed archives
FROM alpine:3.19 AS extractor

RUN apk add --no-cache p7zip

WORKDIR /extract

# Copy model archives and extract
COPY models/custome_ppe.pt.7z.* ./models/
RUN cd models && 7z x custome_ppe.pt.7z.001

# Copy video archives and extract
COPY data/combined-video-no-gap-rooftop.mp4.7z.* ./data/
RUN cd data && 7z x combined-video-no-gap-rooftop.mp4.7z.001

# Copy the existing ppe.pt model (not compressed)
COPY models/ppe.pt ./models/

# Stage 2: Export model to OpenVINO format
FROM docker.io/ultralytics/ultralytics:latest-cpu AS exporter

COPY --from=extractor /extract/models/ppe.pt /tmp/ppe.pt

RUN pip install git+https://github.com/openai/CLIP.git && \
    yolo export model=/tmp/ppe.pt format=openvino task=detect && \
    mkdir -p /exported/ppe/1 && \
    cp /tmp/ppe_openvino_model/*.xml /exported/ppe/1/ppe.xml && \
    cp /tmp/ppe_openvino_model/*.bin /exported/ppe/1/ppe.bin

# Stage 3: Create uploader image with mc CLI
FROM docker.io/minio/mc:latest

WORKDIR /upload

# Copy extracted files from extractor stage
COPY --from=extractor /extract/data/combined-video-no-gap-rooftop.mp4 /upload/data/
COPY --from=extractor /extract/models/custome_ppe.pt /upload/models/
COPY --from=extractor /extract/models/ppe.pt /upload/models/

# Copy exported OpenVINO model
COPY --from=exporter /exported/ppe/1/ /upload/models/ppe/1/

# Copy upload script
COPY data-image/upload.sh /upload/upload.sh
RUN chmod +x /upload/upload.sh

# Default environment variables (can be overridden)
ENV MINIO_ENDPOINT=http://minio:9000
ENV MINIO_ACCESS_KEY=minioadmin
ENV MINIO_SECRET_KEY=minioadmin

ENTRYPOINT ["/bin/sh", "/upload/upload.sh"]
