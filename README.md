# SlidesToText

 -Readme ist KI-Generiert- 

**SlidesToText** ist ein Python-Tool, das PDF-Folien analysiert, den Text extrahiert und für jedes (nicht doppelte) Bild eine automatische Bildbeschreibung generiert. Die Ausgabe ist eine angereicherte Textdatei.

Es gibt zwei Versionen:
- **API-Version** (`slidesToText-API.py`): Nutzt Google Gemini API (funktioniert auf allen Plattformen)
- **MLX-Version** (`slidesToText-MLX.py`): Nutzt lokale MLX-Modelle (nur Apple Silicon Macs)

---

## Voraussetzungen

- Python 3.9 oder neuer (empfohlen: 3.10+)
- [Homebrew](https://brew.sh/) (nur falls Python oder Git fehlen)

**Zusätzlich für MLX-Version:**
- Apple Silicon Mac (M1/M2/M3/M4)

**Zusätzlich für API-Version:**
- Google AI Studio API Key

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

**Für API-Version (Google Gemini):**
```bash
pip install -r requirements-API.txt
```

**Für MLX-Version (Apple Silicon):**
```bash
pip install -r requirements-MLX.txt
```

---

## API-Konfiguration (nur für API-Version)

Für die API-Version benötigst du einen Google AI Studio API Key:

1. Gehe zu [Google AI Studio](https://aistudio.google.com/)
2. Erstelle einen API Key
3. Kopiere die `.env.example` Datei zu `.env`:
   ```bash
   cp .env.example .env
   ```
4. Öffne die `.env` Datei und ersetze `your-google-api-key-here` mit deinem echten API Key

**Alternativ** kannst du den API Key als Umgebungsvariable setzen:
```bash
export GOOGLE_API_KEY="dein-api-key-hier"
```

Alternativ kannst du den API Key direkt im Code in der `_configure_gemini()` Funktion eintragen.

---

## Nutzung

**API-Version (Google Gemini):**
```bash
python3 slidesToText-API.py <input.pdf>
```

**MLX-Version (Apple Silicon):**
```bash
python3 slidesToText-MLX.py <input.pdf>
```

Beispiel:
```bash
python3 slidesToText-API.py TestPDF1.pdf
```

Die Ausgabe-Datei wird automatisch als  
`outcome_<Datum>_<Uhrzeit>.txt`  
im aktuellen Verzeichnis erstellt.

---

## Was macht das Tool?

1. **Text-Extraktion:** Extrahiert den Text-Layer aus dem PDF
2. **Text-Bereinigung:** 
   - Erkennt und entfernt automatisch wiederholte Kopfzeilen (mehrzeilig möglich)
   - Erkennt und entfernt wiederholte Fußzeilen
   - Entfernt mehrfache Leerzeilen
   - Entfernt aufeinanderfolgende identische Zeilen
3. **Bild-Extraktion:** Extrahiert alle Bilder (doppelte werden erkannt und ignoriert)
4. **Platzhalter:** Fügt Platzhalter für die Bilder an der passenden Stelle im Text ein
5. **Bildbeschreibung:** 
   - **API-Version:** Nutzt Google Gemini 1.5 Flash 8B
   - **MLX-Version:** Nutzt lokales Qwen2.5-VL-7B-Instruct-4bit
6. **Zusammenführung:** Ersetzt die Platzhalter durch die Bildbeschreibungen
7. **Formatierung:** Optimiert die finale Textformatierung (optional)
8. **Cleanup:** Löscht temporäre Bilddateien automatisch

---

## Unterschiede zwischen den Versionen

| Feature | API-Version | MLX-Version |
|---------|-------------|-------------|
| **Plattform** | Alle | Nur Apple Silicon |
| **Internet** | Erforderlich | Nicht erforderlich |
| **Kosten** | Pro API-Aufruf | Einmalig (Hardware) |
| **Geschwindigkeit** | Schnell | Mittel |
| **Privatsphäre** | Daten an Google | Vollständig lokal |
| **Modell** | Gemini 1.5 Flash 8B | Qwen2.5-VL-7B |
| **Setup** | API Key erforderlich | Modelle werden automatisch geladen |

---

## Hinweise

- **Doppelte Bilder** (z.B. Logos auf jeder Seite) werden erkannt und nicht verarbeitet
- Die Bildbeschreibung erfolgt auf Deutsch
- Für große PDFs oder viele Bilder kann die Verarbeitung einige Minuten dauern
- **MLX-Version:** Benötigt ausreichend RAM und GPU-Speicher (Apple Silicon empfohlen)
- **API-Version:** Benötigt stabile Internetverbindung
- Kopfzeilen werden automatisch erkannt und entfernt (auch mehrzeilig, siehe Konsolenausgabe)

---

## Optionale Textformatierung

Beide Versionen können den finalen Text optional nachformatieren:

- **API-Version:** Nutzt Google Gemini für die Formatierung
- **MLX-Version:** Nutzt lokales Qwen1.5-1.8B-Chat-4bit (experimentell)

Die Formatierung kann in der jeweiligen `main()` Funktion aktiviert/deaktiviert werden.

---

## Fehlerbehebung

**Allgemein:**
- **ImportError:** Prüfe, ob alle Abhängigkeiten installiert sind und die virtuelle Umgebung aktiv ist

**API-Version:**
- **API Key Fehler:** Stelle sicher, dass `GOOGLE_API_KEY` gesetzt ist oder im Code eingetragen wurde
- **Netzwerkfehler:** Prüfe die Internetverbindung

**MLX-Version:**
- **Out of Memory:** Reduziere die PDF-Größe oder die Anzahl der Bilder
- **MLX Import Error:** Stelle sicher, dass du einen Apple Silicon Mac verwendest

---

## Autor

Konrad Czernohous  
2025