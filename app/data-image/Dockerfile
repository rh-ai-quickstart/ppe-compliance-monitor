# Dockerfile for ppe-compliance-monitor-data
# Contains video and model files for the PPE compliance monitor

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

# Stage 2: Create minimal final image with extracted files
FROM busybox:1.36

WORKDIR /

# Copy extracted files to final locations
COPY --from=extractor /extract/data/combined-video-no-gap-rooftop.mp4 /data/
COPY --from=extractor /extract/models/custome_ppe.pt /models/
COPY --from=extractor /extract/models/ppe.pt /models/

# Set read permissions
RUN chmod -R 644 /data/* /models/*

# Default command just lists the contents. Output looks like this:
#PPE Compliance Monitor Data Container
#--- /data ---
#-rw-r--r--    1 root  root  500M  combined-video-no-gap-rooftop.mp4
#--- /models ---
#-rw-r--r--    1 root  root  150M  custome_ppe.pt
#-rw-r--r--    1 root  root  100M  ppe.pt
CMD ["sh", "-c", "echo 'PPE Compliance Monitor Data Container' && echo '--- /data ---' && ls -lh /data && echo '--- /models ---' && ls -lh /models"]

