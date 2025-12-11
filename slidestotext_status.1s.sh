#!/bin/bash
# xbar/BitBar Plugin: zeigt SlidesToText-Status aus /tmp/slidestotext_status.txt
FILE="/tmp/slidestotext_status.txt"

if [ ! -f "$FILE" ]; then
  echo "---"
  echo "Keine Statusdatei gefunden."
  exit 0
fi

head -n 1 "$FILE"
echo "---"
tail -n +2 "$FILE"
