podman run --rm -v "$PWD:/work" -w /work alpine:3.19 sh -lc \
	"apk add --no-cache p7zip >/dev/null && 7z a -mx=1 -v100m custome_ppe.pt.7z custome_ppe.pt"
