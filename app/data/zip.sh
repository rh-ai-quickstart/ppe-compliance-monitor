podman run --rm -v "$PWD:/work" -w /work alpine:3.19 sh -lc \
  "apk add --no-cache p7zip >/dev/null && 7z a -mx=1 -v100m combined-video-no-gap-rooftop.mp4.7z combined-video-no-gap-rooftop.mp4"