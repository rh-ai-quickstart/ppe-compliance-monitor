FROM --platform=linux/amd64 registry.access.redhat.com/ubi9/python-311:1-77

USER 0
RUN dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    dnf install -y mesa-libGL glib2 && \
    dnf clean all
USER 1001

WORKDIR /app/backend

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

WORKDIR /app

# Model and video files are NOT included in this image.
# For OpenShift/Kubernetes: Files are downloaded from MinIO to a PVC by init container.
# For local development: Use podman-compose which mounts files via volumes or downloads from MinIO.

ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface
ENV HF_HOME=/tmp/.cache/huggingface
ENV XDG_CACHE_HOME=/tmp/.cache
ENV MPLCONFIGDIR=/tmp/.cache/matplotlib
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN mkdir -p /tmp/.cache/huggingface /tmp/.cache/matplotlib /tmp/Ultralytics && \
    chmod 777 /tmp/.cache/huggingface /tmp/.cache/matplotlib /tmp/Ultralytics

EXPOSE 8888

CMD ["python", "/app/backend/app.py"]
