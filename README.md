# SlidesToText WIP

KI-Generiert: 

**SlidesToText** ist ein Python-Tool, das PDF-Folien analysiert, den Text extrahiert und für jedes (nicht doppelte) Bild eine automatische Bildbeschreibung generiert. Die Ausgabe ist eine angereicherte Textdatei.

---

## Voraussetzungen

- Python 3.9 oder neuer (empfohlen: 3.10+)
- macOS (getestet), Linux sollte ebenfalls funktionieren?

---

## Installation

1. **Repository klonen**

```bash
git clone <REPO-URL>
cd SlidesToText
```

2. **Virtuelle Umgebung anlegen (empfohlen)**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Abhängigkeiten installieren**

```bash
pip install -r requirements.txt
```

Falls keine `requirements.txt` vorhanden ist, installiere die Pakete manuell:

```bash
pip install pymupdf pillow mlx-vlm
```

> **Hinweis:**  
> Für MLX-VLM benötigst du ein Apple Silicon Mac (M1/M2/M3).  
> Für andere Plattformen ggf. Alternativen nutzen.

---

## Nutzung

```bash
python3 slidesToText.py <input.pdf>
```

Beispiel:

```bash
python3 slidesToText.py TestPDF1.pdf
```

Die Ausgabe-Datei wird automatisch als  
`outcome_<Datum>_<Uhrzeit>.txt`  
im aktuellen Verzeichnis erstellt.

---

## Was macht das Tool momentan?

1. Extrahiert den Text-Layer aus dem PDF.
2. Extrahiert alle Bilder (doppelte werden erkannt und ignoriert).
3. Fügt Platzhalter für die Bilder in den Text ein.
4. Erstellt für jedes eindeutige Bild eine Bildbeschreibung mit MLX-VLM.
5. Ersetzt die Platzhalter durch die Bildbeschreibungen.
6. Gibt das Ergebnis als Textdatei aus.
7. Löscht temporäre Bilddateien automatisch.

---

## Hinweise

- **Doppelte Bilder** (z.B. Logos auf jeder Seite) werden erkannt und nicht verarbeitet.
- Die Bildbeschreibung erfolgt auf Deutsch.
- Für große PDFs oder viele Bilder kann die Verarbeitung einige Minuten dauern.
- Die Bildbeschreibung benötigt ausreichend RAM und ggf. GPU-Speicher (Apple Silicon empfohlen).

---

## Autor

Konrad Czernohous  
2025
