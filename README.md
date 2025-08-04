# SlidesToText WIP

KI-Generiert: 

**SlidesToText** ist ein Python-Tool, das PDF-Folien analysiert, den Text extrahiert und für jedes (nicht doppelte) Bild eine automatische Bildbeschreibung generiert. Die Ausgabe ist eine angereicherte Textdatei.

---

## Voraussetzungen

- Python 3.9 oder neuer (empfohlen: 3.10+)
- macOS (getestet, Apple Silicon empfohlen)
- [Homebrew](https://brew.sh/) (nur falls Python oder Git fehlen)
- MLX und MLX-VLM benötigen ein Apple Silicon Mac (M1/M2/M3)

---

## Vorbereitung (falls Python oder Git fehlen)

Installiere Homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Installiere Python und Git:

```bash
brew install python git
```

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
pip install pymupdf pillow mlx mlx-vlm
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
2. Erkennt und entfernt automatisch wiederholte Kopfzeilen (mehrzeilig möglich).
3. Extrahiert alle Bilder (doppelte werden erkannt und ignoriert).
4. Fügt Platzhalter für die Bilder an der passenden Stelle im Text ein.
5. Erstellt für jedes eindeutige Bild eine Bildbeschreibung mit MLX-VLM.
6. Ersetzt die Platzhalter durch die Bildbeschreibungen.
7. Gibt das Ergebnis als Textdatei aus.
8. Löscht temporäre Bilddateien automatisch.

---

## Hinweise

- **Doppelte Bilder** (z.B. Logos auf jeder Seite) werden erkannt und nicht verarbeitet.
- Die Bildbeschreibung erfolgt auf Deutsch.
- Für große PDFs oder viele Bilder kann die Verarbeitung einige Minuten dauern.
- Die Bildbeschreibung benötigt ausreichend RAM und GPU-Speicher (Apple Silicon empfohlen).
- Kopfzeilen werden automatisch erkannt und entfernt (auch mehrzeilig, siehe Konsolenausgabe).

---

## Fehlerbehebung

- **Out of Memory:**  
  Reduziere die PDF-Größe oder die Anzahl der Bilder.  
  Stelle sicher, dass keine anderen speicherintensiven Programme laufen.
- **ImportError:**  
  Prüfe, ob alle Abhängigkeiten installiert sind und die virtuelle Umgebung aktiv ist.

---

## Autor

Konrad Czernohous  
2025
