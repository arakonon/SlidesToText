# SlidesToText
2. System-Pakete installieren
brew update
brew install python@3.11      # Homebrew-Python 3.11
brew install poppler          # für pdftotext
brew install ocrmypdf         # PDF → Bilder → Tesseract-OCR
brew install tesseract        # reine OCR-Engine
brew install ghostscript      # PDF-Backend für ocrmypdf
Hinweis: Mit Homebrew-Python wird /opt/homebrew/bin/python3 verfügbar.
Prüfen:
which python3
# → /opt/homebrew/bin/python3
python3 --version
# → Python 3.11.x
3. Virtuelle Umgebung einrichten
Projektordner betreten, vorhandene venv entfernen (falls vorhanden):
rm -rf venv
Neue venv erstellen und aktivieren:
python3 -m venv venv
source venv/bin/activate
Prüfen, dass venv-Python korrekt ist:
which python3
# → …/SlidesToText/venv/bin/python3
python3 --version
# → Python 3.11.x
4. Pip-Konfiguration und Abhängigkeiten installieren
Homebrew-Pythons unter macOS aktivieren PEP 668, daher brauchen wir in der venv das Flag --break-system-packages:
# pip, setuptools, wheel aktualisieren
python3 -m pip install --upgrade --break-system-packages pip setuptools wheel

# Projekt-Dependencies installieren
pip install --break-system-packages \
  PyMuPDF \
  pdf2image \
  pytesseract \
  easyocr \
  transformers \
  pillow \
  openai