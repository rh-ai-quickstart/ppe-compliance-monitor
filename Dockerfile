# Use the specified Red Hat Universal Base Image (UBI) for Python 3.11
FROM --platform=linux/amd64 registry.access.redhat.com/ubi9/python-311:1-77 AS backend

# Set the working directory for the backend
WORKDIR /app/backend

# Copy all backend files
COPY app/backend/ .

# Install backend dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Use UBI-based Node.js image for frontend build
FROM --platform=linux/amd64 registry.access.redhat.com/ubi9/nodejs-18:1-62 AS frontend-build

# Set the working directory for the frontend
WORKDIR /app/frontend

# Copy all frontend files
COPY app/frontend/ .

# Install frontend dependencies
USER 0
RUN npm install && chown -R 1001:0 /app/frontend


# Add missing babel plugin


# Set permissions for npm cache
RUN mkdir -p /opt/app-root/src/.npm && \
    chown -R 1001:0 /opt/app-root/src/.npm && \
    chmod -R 775 /opt/app-root/src/.npm
USER 1001

ENV DISABLE_ESLINT_PLUGIN=true
RUN npm run build

# Final stage
FROM --platform=linux/amd64 registry.access.redhat.com/ubi9/python-311:1-77

# Set the working directory
WORKDIR /app

# Install system dependencies for OpenCV
USER 0
RUN dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    dnf install -y mesa-libGL glib2 nodejs npm && \
    dnf clean all

# Copy backend from the backend stage
COPY --from=backend /app/backend /app/backend

# Copy models and data into the image
RUN mkdir -p /app/models /app/data
COPY app/models/*.pt /app/models/
COPY app/data/*.mp4 /app/data/

# Copy built frontend from the frontend-build stage
COPY --from=frontend-build /app/frontend/build /app/frontend/build

# Install backend dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Install serve to serve the frontend
RUN npm install -g serve

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface
ENV HF_HOME=/tmp/.cache/huggingface
ENV XDG_CACHE_HOME=/tmp/.cache
ENV MPLCONFIGDIR=/tmp/.cache/matplotlib
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# Create cache directories
RUN mkdir -p /tmp/.cache/huggingface /tmp/.cache/matplotlib /tmp/Ultralytics && \
    chmod 777 /tmp/.cache/huggingface /tmp/.cache/matplotlib /tmp/Ultralytics

# Copy the start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose the ports the app runs on
EXPOSE 8888 3000

# Change ownership of the app directory
RUN chown -R 1001:0 /app && \
    chmod -R g+w /app

# Switch back to non-root user
USER 1001

# Debug: List contents of /app/backend and /app/frontend
RUN echo "Contents of /app/backend:" && ls -la /app/backend && \
    echo "Contents of /app/frontend/build:" && ls -la /app/frontend/build

# Command to run the start script
CMD ["/app/start.sh"]