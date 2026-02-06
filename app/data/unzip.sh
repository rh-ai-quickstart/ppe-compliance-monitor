

podman run --rm -v "$PWD:/work" -w /work alpine:3.19 sh -lc \
  "apk add --no-cache p7zip >/dev/null && 7z x combined-video-no-gap-rooftop.mp4.7z.001"