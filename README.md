# SlidesToText
PDF rein, bereinigter Text raus – inkl. KI-Bildbeschreibungen.

- API: `slidesToText-API.py` (Google Gemini, läuft überall)
- MLX: `slidesToText-MLX.py` (lokal auf Apple Silicon)

## Quick Start
```bash
git clone <REPO-URL> && cd SlidesToText
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-API.txt    # oder: requirements-MLX.txt
python3 slidesToText-API.py deine-folien.pdf
```

## API Key (nur API)
```bash
cp .env.example .env
echo 'GOOGLE_API_KEY=dein-key' >> .env   # oder: export GOOGLE_API_KEY=dein-key
```

## Ablauf
1) Text extrahieren und säubern (Kopf/Fußzeilen, Duplikate)  
2) Bilder finden, Duplikate entfernen  
3) KI beschreibt die Bilder  
4) Text + Bildbeschreibungen zusammenführen → `outcome_<Datum>_<Zeit>.txt`  
API-Modus: `images/` öffnet sich, ungewollte Bilder löschen, Enter drücken.

## Low-Power-Mode (macOS)
- Default: `--low-power auto` → erkennt aktiven Batteriesparmodus, schaltet ihn für die Laufzeit aus und stellt ihn danach wieder her. Wenn er aus ist, passiert nichts.  
- Weitere Modi: `keep` (nicht anfassen), `on`, `off`. Mit `--restore-low-power` kannst du in den Modi `on/off` den ursprünglichen Zustand zurückholen.
- Beispiel:  
  ```bash
  python3 slidesToText-MLX.py --low-power auto dein.pdf
  ```
- Keine Passwortabfrage gewuenscht? In `/etc/sudoers` (per `sudo visudo`) z. B.:  
  ```
  NUTZERNAME ALL=(root) NOPASSWD: /usr/bin/pmset -a lowpowermode 0, /usr/bin/pmset -a lowpowermode 1
  ```
  (Nutzername anpassen.)

## xbar-Status (optional, macOS)
```bash
chmod +x "/Users/konrad/Desktop/Programmier Stuff/SlidesToText/slidestotext_status.1s.sh"
ln -sf "/Users/konrad/Desktop/Programmier Stuff/SlidesToText/slidestotext_status.1s.sh" \
  "$HOME/Library/Application Support/xbar/plugins/slidestotext_status.1s.sh"
```
Dann xbar Refresh; während des Laufs steht der Status in `/tmp/slidestotext_status.txt`.

## Tipps
- Viele gleiche Logos? Werden automatisch ignoriert.
- Schlechte Bilder? In `images/` löschen, dann Enter.
- Alte Ergebnisse landen in `Legacy Outcomes/`.

Made by Konrad Czernohous • 2025
