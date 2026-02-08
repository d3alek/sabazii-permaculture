#!/bin/bash
# Sync Google Doc → local markdown via DOCX export + pandoc
# Usage: ./sync-gdoc.sh
#
# Requires: pandoc, curl
# Works with any publicly shared Google Doc (no API keys needed).

set -euo pipefail

DOC_ID="1WO1MAfWaJisfWueIRMQ0X0OPMheXIu2b2l3al_QKhu4"
OUTPUT="final-design.md"
TMPDIR=$(mktemp -d)
DOCX_FILE="$TMPDIR/export.docx"

echo "=== Google Doc → Markdown Sync ==="
echo ""
echo "Step 1: Downloading DOCX from Google Docs..."
curl -sL "https://docs.google.com/document/d/${DOC_ID}/export?format=docx" -o "$DOCX_FILE"
file "$DOCX_FILE" | grep -q "Microsoft Word" || { echo "Error: Downloaded file is not a DOCX (is the doc publicly shared?)"; exit 1; }

echo "Step 2: Converting to Markdown with pandoc..."
pandoc -f docx -t gfm --wrap=none "$DOCX_FILE" -o "$OUTPUT"

# Add Hugo front matter
TMPFILE=$(mktemp)
cat > "$TMPFILE" << 'FRONTMATTER'
---
title: "Еко-селище Сабазий — Пермакултурен дизайн"
date: 2026-02-08
draft: false
---

FRONTMATTER
cat "$OUTPUT" >> "$TMPFILE"
mv "$TMPFILE" "$OUTPUT"

echo "Step 3: Verifying encoding..."
NONASCII=$(python3 -c "print(sum(1 for b in open('$OUTPUT','rb').read() if b > 127))")
echo "  Non-ASCII bytes: $NONASCII (should be >0 for Cyrillic)"

if [ "$NONASCII" -eq 0 ]; then
    echo "ERROR: No Cyrillic characters found — encoding problem!"
    exit 1
fi

LINES=$(wc -l < "$OUTPUT")
echo ""
echo "Done! $OUTPUT updated ($LINES lines, $NONASCII non-ASCII bytes)"
echo ""
echo "To push to GitHub:"
echo "  cd /path/to/sabazii-permaculture"
echo "  cp /path/to/$OUTPUT content/final-design.md"
echo "  git add content/final-design.md && git commit -m 'Sync from Google Docs' && git push"

rm -rf "$TMPDIR"
