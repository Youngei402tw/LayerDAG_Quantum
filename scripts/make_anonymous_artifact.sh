#!/usr/bin/env bash
# Build a double-blind artifact archive from the committed tree (no .git history, no author metadata).
# Usage: bash scripts/make_anonymous_artifact.sh [OUTPUT_TARBALL]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out="${1:-qai_anonymous_artifact.tar.gz}"
tmp="$(mktemp -d)"
git archive --format=tar HEAD | tar -x -C "$tmp"
# drop archived/obsolete manuscripts and build products from the anonymous copy
rm -rf "$tmp/paper/archive" "$tmp"/paper/*.aux "$tmp"/paper/*.log "$tmp"/paper/*.bbl "$tmp"/paper/*.blg
# fail if identifying strings remain (extend IDENTIFIERS with every author name/handle/email before use)
IDENTIFIERS="${IDENTIFIERS:-Youngei402tw|YANG-CHE|yangche|u\.nus\.edu|gmail\.com}"
if grep -rIEn "$IDENTIFIERS" "$tmp"; then echo "identifying strings found; aborting" >&2; exit 1; fi
# strip PDF metadata author fields if exiftool/qpdf are present (best effort)
command -v exiftool >/dev/null && find "$tmp" -name '*.pdf' -exec exiftool -q -overwrite_original -Author= -Creator= {} + || true
tar -czf "$out" -C "$tmp" .
rm -rf "$tmp"
sha256sum "$out"
