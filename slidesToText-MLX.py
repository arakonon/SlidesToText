#!/usr/bin/env python3
import sys
import argparse
import os
import fitz                   
from PIL import Image
from mlx_vlm import load as load_vlm, generate as generate_vlm
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from mlx_lm import load as load_lm, generate as generate_lm
import pytesseract, cv2, numpy as np
from PIL import Image
import shutil
import datetime
import hashlib
from collections import Counter
import re
import subprocess
import glob
import platform
import threading
import time
import csv



def extract_text_layer(pdf_path):
    doc = fitz.open(pdf_path)
    return [page.get_text() for page in doc]  # Liste: ein Eintrag pro Seite

def extract_images(pdf_path, img_folder="images"):
    import collections
    os.makedirs(img_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    img_files = []
    img_placeholders = collections.defaultdict(list)  # Seite -> [Platzhalter]
    hash_to_paths = collections.defaultdict(list)

    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            data = base["image"]
            ext = base["ext"]
            fname = f"page{page_num+1}_{img_index}.{ext}"
            path = os.path.join(img_folder, fname)
            with open(path, "wb") as f:
                f.write(data)
            h = image_hash(path)
            hash_to_paths[h].append((path, page_num))

    for paths in hash_to_paths.values():
        # Ab wievielen Bilder, Bild entfernen <---------------------
        if len(paths) >= 4:
            for p, _ in paths:
                print(f"Doppeltes Bild erkannt (insgesamt {len(paths)} Vorkommen) und entfernt: {os.path.basename(p)}\n")
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
            continue
        # Unterhalb der Löschschwelle nur ein Exemplar beschreiben, aber auf allen Seiten den gleichen Platzhalter setzen.
        keep_path, _ = paths[0]
        img_files.append(keep_path)
        placeholder = f"[IMG:{os.path.basename(keep_path)}]"
        for _, page_num in paths:
            img_placeholders[page_num].append(placeholder)
    return img_files, img_placeholders

def caption_images(img_files, model_path="mlx-community/Qwen3-VL-8B-Instruct-4bit", set_phase_cb=None):
    print(f"{len(img_files)} Bilder werden beschrieben...\n")
    model, processor = load_vlm(model_path)
    cfg = load_config(model_path)
    bildNr = 0
    captions = {}
    for img in img_files:
        if set_phase_cb:
            set_phase_cb(f"Bilder {bildNr+1}/{len(img_files)}")
        pil_img = [Image.open(img).convert("RGB").resize((512, 512))]
        prompt = "Beschreibe dieses Bild auf Deutsch. Wenn es sich um eine Fotografie oder Szene handelt, beschreibe in maximal 2 kurzen Sätzen. Wenn es sich um ein Diagramm, eine Skizze oder eine schematische Darstellung handelt, beschreibe das Bild sehr genau und interpretiere es. Wenn das Bild nur Text enthält, gib nur den Text wieder. Wenn Teile des Bildes nicht erkennbar sind, weise darauf hin."
        prompt_fmt = apply_chat_template(processor, cfg, prompt, num_images=len(pil_img))
        cap = generate_vlm(model, processor, prompt_fmt, pil_img, verbose=False)
        captions[img] = cap.text.strip()
        bildNr += 1
        print("\nBild Nr.", bildNr, "beschrieben\n")
    return captions

def merge_text(text_with_place, captions):
    """
    Ersetzt Platzhalter durch Bildbeschreibungen und entfernt übrig gebliebene Platzhalter
    """
    enriched = text_with_place
    for img in captions:
        placeholder = f"[IMG:{os.path.basename(img)}]"
        enriched = enriched.replace(placeholder, f"BILD: [{captions[img]}]\n")
    # Entferne alle übrig gebliebenen Platzhalter (für gelöschte Bilder)
    enriched = re.sub(r'\[IMG:[^\]]+\]\n?', '', enriched)
    return enriched

def insert_placeholders(text_layer_list, img_placeholders, existing_imgs):
    """
    Fügt nur Platzhalter für tatsächlich vorhandene Bilder ein
    """
    existing_basenames = {os.path.basename(img) for img in existing_imgs}
    result = []
    for i, text in enumerate(text_layer_list):
        filtered_placeholders = []
        for placeholder in img_placeholders.get(i, []):
            if placeholder.startswith("[IMG:") and placeholder.endswith("]"):
                img_name = placeholder[5:-1]
                if img_name in existing_basenames:
                    filtered_placeholders.append(placeholder)
        placeholders_text = "\n".join(filtered_placeholders)
        result.append(text + ("\n" + placeholders_text if placeholders_text else ""))
    return "\n\n".join(result)

def image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def remove_repeated_headers_auto(text_layer_list, min_count=5, max_header_lines=5):
    print("Suche nach wiederholten Kopfzeilen ...")
    candidates = []
    for n in range(1, max_header_lines+1):
        blocks = [
            "\n".join(page.splitlines()[:n])
            for page in text_layer_list if len(page.splitlines()) >= n
        ]
        block_counts = Counter(blocks)
        for block, count in block_counts.items():
            if count >= min_count:
                print(f"Kandidat für Kopfzeile ({n} Zeilen, {count} Vorkommen):\n---\n{block}\n---\n")
                candidates.append((n, block, count))
    if not candidates:
        print("Keine Kopfzeile erkannt.")
        return text_layer_list
    candidates.sort(key=lambda x: (x[0], x[2]), reverse=True)
    header_lines, header_block, _ = candidates[0]
    print(f"Entferne Kopfzeile ({header_lines} Zeilen):\n---\n{header_block}\n---\n")
    cleaned = []
    removed_count = 0
    first_found = False
    for page in text_layer_list:
        lines = page.splitlines()
        block = "\n".join(lines[:header_lines])
        if block == header_block:
            if not first_found:
                first_found = True  # Erste Instanz bleibt erhalten
            else:
                lines = lines[header_lines:]
                removed_count += 1
        cleaned.append("\n".join(lines))
    print(f"Kopfzeile auf {removed_count} Seiten entfernt.\n")
    return cleaned

def remove_repeated_footers_auto(text_layer_list, min_count=5, max_footer_lines=5):
    print("Suche nach wiederholten Fußzeilen ...")
    candidates = []
    for n in range(1, max_footer_lines+1):
        blocks = [
            "\n".join(mask_footer_line(line) for line in page.splitlines()[-n:])
            for page in text_layer_list if len(page.splitlines()) >= n
        ]
        block_counts = Counter(blocks)
        for block, count in block_counts.items():
            if count >= min_count:
                print(f"Kandidat für Fußzeile ({n} Zeilen, {count} Vorkommen):\n---\n{block}\n---\n")
                candidates.append((n, block, count))
    if not candidates:
        print("Keine Fußzeile erkannt.")
        return text_layer_list
    candidates.sort(key=lambda x: (x[0], x[2]), reverse=True)
    footer_lines, footer_block, _ = candidates[0]
    print(f"Entferne Fußzeile ({footer_lines} Zeilen):\n---\n{footer_block}\n---\n")
    cleaned = []
    removed_count = 0
    first_found = False
    for page in text_layer_list:
        lines = page.splitlines()
        block = "\n".join(mask_footer_line(line) for line in lines[-footer_lines:])
        if block == footer_block:
            if not first_found:
                first_found = True  # Erste Instanz bleibt erhalten
            else:
                lines = lines[:-footer_lines]
                removed_count += 1
        cleaned.append("\n".join(lines))
    print(f"Fußzeile auf {removed_count} Seiten entfernt.\n")
    return cleaned

def remove_multiple_blank_lines_per_page(text_layer_list):
    cleaned_pages = []
    for page in text_layer_list:
        # Ersetze mehrere aufeinanderfolgende Leerzeilen durch eine
        cleaned = re.sub(r'\n\s*\n+', '\n', page)
        cleaned_pages.append(cleaned.strip())
    return cleaned_pages

def remove_consecutive_duplicate_lines(text_layer_list):
    """
    Entfernt unmittelbar aufeinanderfolgende identische Zeilen (z. B. ständig wiederholte Folien‑Titel).
    """
    cleaned_pages = []
    for page in text_layer_list:
        new_lines = []
        prev = None
        for line in page.splitlines():
            current = line.strip()
            if current == prev and current != "":
                # überspringe Duplikat
                continue
            new_lines.append(line)
            prev = current
        cleaned_pages.append("\n".join(new_lines))
    return cleaned_pages

def mask_footer_line(line):
    # Entfernt Zahlen und typische Seitenzahl-Muster
    line = re.sub(r'\b\d+\s*/\s*\d+\b', '', line)  # Muster: "9 / 11"
    line = re.sub(r'\bSeite\s*\d+\b', '', line, flags=re.IGNORECASE)
    line = re.sub(r'\d+', '', line)  # Alle Zahlen
    return line.strip()

def format_ocr(text: str) -> str:
    # Versuche Qwen mit Fast-Tokenizer (benötigt KEIN sentencepiece).
    # Wenn das Laden scheitert, wird abgebrochen und der unformatierte Text zurückgegeben.
    try:
        model_format, tok = load_lm(
            "mlx-community/Qwen3-1.7B-4bit",
            tokenizer_config={
                "use_fast": True,
                "trust_remote_code": True,
            },
        )
        # Qwen 3 hat einen großen Kontext, wir nutzen konservativ 32k
        max_ctx_tokens = 32000
        # Kontextbudget zwischen Eingabetext und generiertem Text aufteilen
        reserve_for_system = 512
        available = max_ctx_tokens - reserve_for_system
        if available <= 0:
            print("Warnung: Kontextbudget ist zu klein. Gebe unformatierten Text zurück.")
            return text
        # Etwa halbe-halbe: die Hälfte für Eingabe, die Hälfte für Ausgabe
        max_user_tokens = available // 2
        # Aggressiver chunken: hart deckeln, damit das Modell kleinere Stücke bekommt
        max_user_tokens = min(max_user_tokens, 6000)
        max_new_tokens = available - max_user_tokens
        # Cap, damit das Modell nicht endlos weitergeneriert
        max_new_tokens = min(max_new_tokens, 4096)
    except Exception as e:
        print("Fehler beim Laden von Qwen3-8B-4bit. Gebe unformatierten Text zurück.")
        print(f"Detail: {e}")
        return text

    system = (
        "Du bist ein deutschsprachiger Texteditor. "
        "Du musst die folgenden Formatierungen anwenden. "
        "Gib den gesamten Text vollständig zurück. Nichts weglassen, nichts hinzufügen, nichts umsortieren. "
        "Entferne Zeilenumbrüche mitten im Satz. "
        "Korrigiere Leerzeichen vor Satzzeichen. "
        "Entferne doppelte oder unnötige Leerzeilen. "
        "Strukturiere in sinnvolle Absätze. "
        "Entferne wiederholte Kopf- und Fußzeilen sowie Seitenzahlen. "
        "Ändere keine inhaltlichen Aussagen oder Formulierungen. "
        "Antworte ausschließlich mit dem formatierten Text."
    )

    def run_chunk(chunk_text: str) -> str:
        """Formatiert einen einzelnen Chunk mit dem LLM."""
        if not chunk_text.strip():
            return chunk_text

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": chunk_text.strip()},
        ]

        # Template anwenden, falls vorhanden, sonst Fallback-Prompt
        try:
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = system + "\n\n" + chunk_text.strip() + "\n\n"

        # Anzahl neuer Tokens pro Chunk begrenzen
        # (groß genug, um den Text vollständig neu zu formatieren)
        max_new = max_new_tokens

        try:
            out = generate_lm(
                model_format,
                tok,
                prompt=prompt,
                max_tokens=max_new,
            )
        except Exception as e:
            print("Fehler bei der LLM-Generierung für einen Chunk. Gebe Chunk unverändert zurück.")
            print(f"Detail: {e}")
            return chunk_text
        out = (out or "").strip()
        # Fallback: Wenn das LLM nichts liefert, gib den Rohtext zurück
        if not out:
            print("Hinweis: LLM lieferte leeren Output für einen Chunk – gebe Chunk unverändert zurück.")
            return chunk_text
        # Entferne evtl. ausgegebenes Reasoning im <think>-Block (auch wenn kein </think> vorhanden)
        try:
            out = re.sub(r"<think>.*?(</think>|$)", "", out, flags=re.DOTALL)
        except Exception:
            # Falls das Regex aus irgendeinem Grund scheitert, ignorieren wir es
            pass
        # Optionale Entfernung von Antwort-Tags, falls das Modell <answer>...</answer> nutzt
        out = out.replace("<answer>", "").replace("</answer>", "").strip()
        # Warnungen bei starker Abweichung, aber keine automatische Ersetzung
        if len(out) > len(chunk_text) * 2:
            print("Warnung: LLM-Ausgabe ist viel länger als der Eingabetext.")
        if len(out) < len(chunk_text) * 0.7:
            print("Warnung: LLM-Ausgabe ist deutlich kürzer als der Eingabetext.")
        return out

    text = text.strip()
    if not text:
        return text

    # Versuche, Token-basiert zu chunken, um den Kontext maximal auszunutzen.
    chunks = []
    try:
        user_tokens = tok.encode(text)

        if len(user_tokens) <= max_user_tokens:
            # Passt komplett in einen Kontext → einmal durch das LLM schicken
            return run_chunk(text)

        # Sonst in mehrere Chunks aufteilen, die jeweils sicher in den Kontext passen
        for start in range(0, len(user_tokens), max_user_tokens):
            end = start + max_user_tokens
            sub_ids = user_tokens[start:end]
            sub_text = tok.decode(sub_ids)
            chunks.append(sub_text)
    except Exception:
        # Wenn der Tokenizer hier Probleme macht, auf Zeichen-basierte Aufteilung zurückfallen
        print("Hinweis: Konnte Tokenizer nicht für Chunking verwenden – weiche auf Zeichen-basierte Aufteilung aus.")
        max_chars = 8000  # konservativer Wert, damit der Kontext sicher reicht
        if len(text) <= max_chars:
            return run_chunk(text)
        for start in range(0, len(text), max_chars):
            chunks.append(text[start:start + max_chars])

    # Jetzt alle Chunks nacheinander durch das Modell schicken und wieder zusammensetzen
    results = []
    total_chunks = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        print(f"Verarbeite Chunk {idx}/{total_chunks} (Länge: {len(chunk)} Zeichen)...")
        formatted = run_chunk(chunk)
        results.append(formatted)

    # Mit Leerzeilen trennen, damit Absatzgrenzen zwischen den Chunks erhalten bleiben
    return "\n\n".join(results)


def move_old_outcome_files():
    """
    Verschiebt alle bestehenden outcome_*.txt Dateien in den Legacy Outcomes Ordner
    """
    legacy_folder = "Legacy Outcomes"
    os.makedirs(legacy_folder, exist_ok=True)
    outcome_files = glob.glob("outcome_*.txt")
    if outcome_files:
        print(f"Verschiebe {len(outcome_files)} alte outcome-Datei(en) nach '{legacy_folder}/'...")
        for old_file in outcome_files:
            destination = os.path.join(legacy_folder, old_file)
            if os.path.exists(destination):
                name, ext = os.path.splitext(old_file)
                timestamp = datetime.datetime.now().strftime('%H-%M-%S')
                destination = os.path.join(legacy_folder, f"{name}_{timestamp}{ext}")
            try:
                shutil.move(old_file, destination)
                print(f"{old_file} → {destination}")
            except Exception as e:
                print(f"Fehler beim Verschieben von {old_file}: {e}")
        print()


def open_file_or_folder(path):
    """Öffnet Datei oder Ordner plattformspezifisch"""
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", path], check=True)
        elif platform.system() == "Windows":
            subprocess.run(["explorer", path], check=True)
        elif platform.system() == "Linux":
            subprocess.run(["xdg-open", path], check=True)
        else:
            print(f"Unbekanntes System. Öffne '{path}' manuell.")
            return False
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def close_folder_window(folder_path):
    """Schließt Ordner-Fenster (nur macOS). Gibt Anzahl geschlossener Fenster zurück."""
    if platform.system() != "Darwin":
        return 0
    try:
        abs_path = os.path.abspath(folder_path)
        applescript = f'''
        tell application "Finder"
            try
                set imagePath to POSIX file "{abs_path}" as alias
                set matches to (every window whose target is imagePath)
                set n to count of matches
                repeat with w in matches
                    try
                        close w
                    end try
                end repeat
                return n
            on error errMsg number errNum
                return 0
            end try
        end tell
        '''
        proc = subprocess.run(["osascript", "-e", applescript], check=True, capture_output=True, text=True)
        try:
            return int(proc.stdout.strip()) if proc.stdout.strip() != '' else 0
        except ValueError:
            return 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0


# ---------- Batteriesparmodus ----------
def get_low_power_mode_state():
    """
    Liefert den aktuellen Batteriesparmodus auf macOS als bool oder None, falls nicht ermittelbar/nicht macOS.
    """
    if platform.system() != "Darwin":
        return None
    try:
        proc = subprocess.run(
            ["pmset", "-g"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in proc.stdout.splitlines():
            if "lowpowermode" in line.lower():
                try:
                    return bool(int(line.split()[-1]))
                except Exception:
                    return None
    except Exception:
        return None
    return None


def set_low_power_mode_state(enable):
    """
    Schaltet den Batteriesparmodus via pmset. Versucht erst ohne, dann mit Admin-Dialog (osascript).
    Gibt True bei Erfolg zurück.
    """
    if platform.system() != "Darwin":
        return False
    target = "1" if enable else "0"
    # Reihenfolge: sudo (NOPASSWD-Regel kann greifen), plain pmset, dann AppleScript mit Admin-Prompt.
    commands = [
        ["sudo", "-n", "/usr/bin/pmset", "-a", "lowpowermode", target],
        ["/usr/bin/pmset", "-a", "lowpowermode", target],
        ["osascript", "-e", f'do shell script "/usr/bin/pmset -a lowpowermode {target}" with administrator privileges'],
    ]
    for cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                return True
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return False


# ---------- Statistik/Schätzung ----------
STATS_FILE = "processing_stats.csv"
ETA_MIN_SECONDS = 1
ETA_MAX_SECONDS = 6 * 3600  # 6 Stunden Hardcap
OUTLIER_FACTOR = 3.0        # Faktor um Median, außerhalb dessen Werte ignoriert werden


def load_stats(stats_path=STATS_FILE):
    records = []
    if not os.path.exists(stats_path):
        return records
    try:
        with open(stats_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    est_val = row.get("estimated_sec", "")
                    delta_val = row.get("delta_sec", "")
                    est_num = float(est_val) if (est_val not in (None, "",)) else None
                    delta_num = float(delta_val) if (delta_val not in (None, "",)) else None
                    records.append(
                        {
                            "chars": int(float(row.get("chars", 0) or 0)),
                            "images": int(float(row.get("images", 0) or 0)),
                            "duration_sec": float(row.get("duration_sec", 0) or 0),
                            "estimated_sec": est_num,
                            "delta_sec": delta_num,
                        }
                    )
                except Exception:
                    continue
    except Exception as e:
        print(f"Warnung: Konnte Statistikdatei nicht lesen ({e}).")
    return records


def append_stat(record, stats_path=STATS_FILE):
    fieldnames = ["chars", "images", "duration_sec", "estimated_sec", "delta_sec"]
    try:
        # Bestehende Daten einlesen und mit neuem Record erneut schreiben, damit das Header-Layout konsistent bleibt
        history = load_stats(stats_path=stats_path)
        history.append(record)
        with open(stats_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in history:
                def fmt(v):
                    return "" if v is None else v
                writer.writerow(
                    {
                        "chars": r.get("chars", 0),
                        "images": r.get("images", 0),
                        "duration_sec": r.get("duration_sec", 0),
                        "estimated_sec": fmt(r.get("estimated_sec")),
                        "delta_sec": fmt(r.get("delta_sec")),
                    }
                )
    except Exception as e:
        print(f"Warnung: Konnte Statistik nicht schreiben ({e}).")


def compute_eta_accuracy(history, window=6):
    """
    Liefert die mittlere ETA-Genauigkeit (%) der letzten `window` Läufe:
    100 * (1 - |ETA - Dauer| / Dauer), auf 0..100 begrenzt.
    """
    usable = [
        r for r in history
        if r.get("duration_sec", 0) > 0 and r.get("estimated_sec") not in (None, "")
    ]
    if not usable:
        return None
    usable = usable[-window:]
    acc_vals = []
    for r in usable:
        dur = r.get("duration_sec", 0)
        est = r.get("estimated_sec", None)
        if dur <= 0 or est is None:
            continue
        try:
            acc = 1 - abs(float(est) - float(dur)) / float(dur)
            acc_vals.append(max(0.0, min(acc, 1.0)))
        except Exception:
            continue
    if not acc_vals:
        return None
    return round(sum(acc_vals) / len(acc_vals) * 100, 1)


def estimate_duration_from_history(history, chars, images):
    """
    Schätzt Laufzeit per linearer Regression (Dauer = a*Zeichen + b*Bilder + c),
    fallback auf einfache Mittelwerte.
    """
    if not history:
        return None

    # Ausreißer filtern anhand Median
    durations = [r.get("duration_sec", 0) for r in history if r.get("duration_sec", 0) > 0]
    if not durations:
        return None
    filtered = history
    if len(durations) >= 3:
        med = np.median(durations)
        lower = med / OUTLIER_FACTOR
        upper = med * OUTLIER_FACTOR
        filtered = [r for r in history if lower <= r.get("duration_sec", 0) <= upper and r.get("duration_sec", 0) > 0]
        if not filtered:
            filtered = history  # Fallback, falls alles rausgefiltert wurde
    history = filtered

    A = []
    y = []
    for r in history:
        duration = r.get("duration_sec", 0)
        if duration <= 0:
            continue
        A.append([r.get("chars", 0), r.get("images", 0), 1])
        y.append(duration)

    est = None
    if len(A) >= 2:
        try:
            coeffs, _, _, _ = np.linalg.lstsq(
                np.array(A, dtype=float), np.array(y, dtype=float), rcond=None
            )
            est = float(coeffs[0] * chars + coeffs[1] * images + coeffs[2])
            if est <= 0:
                est = None
        except Exception:
            est = None

    if est is None:
        total_chars = sum(r.get("chars", 0) for r in history)
        total_images = sum(r.get("images", 0) for r in history)
        total_dur = sum(r.get("duration_sec", 0) for r in history)
        sec_per_char = (total_dur / total_chars) if total_chars > 0 else 0
        sec_per_image = (total_dur / total_images) if total_images > 0 else 0
        est = sec_per_char * chars + sec_per_image * images
        if est <= 0:
            return None
    # ETA Grenzen anwenden
    est = max(est, ETA_MIN_SECONDS)
    est = min(est, ETA_MAX_SECONDS)
    return est


def clamp_eta(seconds):
    try:
        val = float(seconds)
    except Exception:
        return None
    val = max(val, ETA_MIN_SECONDS)
    val = min(val, ETA_MAX_SECONDS)
    return val


# ---------- xbar Handling ----------
def ensure_xbar_running():
    """
    Startet xbar, falls es nicht läuft. Gibt True zurück, wenn xbar bereits lief,
    sonst False (dann wurde es von uns gestartet).
    """
    if platform.system() != "Darwin":
        return True  # auf Nicht-macOS nichts tun
    try:
        was_running = subprocess.run(
            ["pgrep", "-x", "xbar"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    except Exception:
        was_running = False
    if not was_running:
        try:
            subprocess.run(["open", "-ga", "xbar"], check=False)
        except Exception:
            pass
    return was_running


def stop_xbar_if_started(was_running):
    """
    Beendet xbar nur, wenn wir es selbst gestartet haben.
    """
    if platform.system() != "Darwin":
        return
    if was_running:
        return
    try:
        subprocess.run(
            ["osascript", "-e", 'tell application "xbar" to quit'],
            check=False,
        )
    except Exception:
        pass


# ---------- Status-Tracking (z. B. für xbar) ----------
def start_status_timer(label="SlidesToText-MLX", interval=1, start_ts=None):
    """
    Schreibt Status, verstrichene Zeit und ETA in /tmp/slidestotext_status.txt.
    Liefert set_phase(text), set_estimated_total(seconds), set_start_time(ts) und stop() zurück.
    """
    start_ts = start_ts or time.time()
    status = {"phase": "Start", "est_total": None, "start_ts": start_ts}
    stop_event = threading.Event()
    status_file = "/tmp/slidestotext_status.txt"

    def set_phase(text):
        status["phase"] = text

    def set_estimated_total(seconds):
        if seconds is None:
            status["est_total"] = None
            return
        try:
            status["est_total"] = max(float(seconds), 0)
        except Exception:
            status["est_total"] = None

    def set_start_time(ts):
        try:
            status["start_ts"] = float(ts)
        except Exception:
            pass

    def writer():
        while not stop_event.is_set():
            elapsed = int(time.time() - status["start_ts"])
            try:
                line1 = f"{status['phase']}"
                remaining = None
                if status["est_total"] is not None:
                    remaining = int(status["est_total"] - elapsed)  # darf ins Minus laufen
                    line1 = f"{status['phase']} · ETA {remaining}s"
                with open(status_file, "w") as f:
                    f.write(f"{line1}\n")
                    f.write("---\n")
                    f.write(f"{status['phase']}\n")
                    if remaining is not None:
                        f.write(f"Verbleibend ~{remaining}s (gesamt ~{int(status['est_total'])}s)\n")
                # Leichtes Flush-Intervall
            except Exception:
                pass
            time.sleep(interval)

    t = threading.Thread(target=writer, daemon=True)
    t.start()

    def stop():
        stop_event.set()
        t.join(timeout=2)
        try:
            with open(status_file, "w") as f:
                f.write(f"⏱ Fertig\n")
                f.write("---\n")
                f.write("Fertig\n")
        except Exception:
            pass

    return set_phase, set_estimated_total, set_start_time, stop


def main():
    parser = argparse.ArgumentParser(description="Extrahiert Text und Bilder aus PDFs und reichert sie mit MLX-Modellen an.")
    parser.add_argument("input_pdf", help="Pfad zur PDF-Datei")
    parser.add_argument(
        "--low-power",
        choices=["auto", "keep", "on", "off"],
        default="auto",
        help="Batteriesparmodus (macOS): auto=falls an, temporär aus und am Ende wieder an; keep=unverändert; on/off erzwingt Zustand.",
    )
    parser.add_argument(
        "--restore-low-power",
        action="store_true",
        help="Ursprünglichen Batteriesparmodus nach dem Lauf wiederherstellen.",
    )
    args = parser.parse_args()
    pdf_in = args.input_pdf
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    stats_history = load_stats()
    script_start_ts = time.time()
    processing_start_ts = None
    est_total_in_use = None
    low_power_changed = False
    initial_low_power = get_low_power_mode_state()
    was_running = ensure_xbar_running()
    set_phase, set_estimated_total, set_start_time, stop_status = start_status_timer(start_ts=script_start_ts)
    set_phase("Starte")

    try:
        if args.low_power != "keep":
            if platform.system() != "Darwin":
                print("Batteriesparmodus-Umschaltung wird nur auf macOS unterstützt.\n")
            else:
                if args.low_power == "auto":
                    if initial_low_power:
                        print("Batteriesparmodus ist aktiv → schalte für die Laufzeit aus...\n")
                        success = set_low_power_mode_state(False)
                        if success:
                            low_power_changed = True
                            print("Batteriesparmodus wurde temporär ausgeschaltet.\n")
                        else:
                            print("Konnte Batteriesparmodus nicht ausschalten (Admin-Rechte erforderlich?).\n")
                    else:
                        print("Batteriesparmodus ist bereits aus, nichts zu tun.\n")
                else:
                    target_state = args.low_power == "on"
                    print(f"Schalte Batteriesparmodus {'ein' if target_state else 'aus'}...\n")
                    success = set_low_power_mode_state(target_state)
                    if success:
                        low_power_changed = initial_low_power is not None and initial_low_power != target_state
                        print("Batteriesparmodus umgeschaltet.\n")
                    else:
                        print("Konnte Batteriesparmodus nicht umschalten (Admin-Rechte erforderlich?).\n")

        # Alte outcome-Dateien aufräumen
        move_old_outcome_files()

        # Erzeuge Dateinamen mit Datum und Uhrzeit
        now = datetime.datetime.now()
        out_txt = f"outcome_{now.strftime('%d.%m.%y_%H:%M')}.txt"

        # 1. Text-Layer extrahieren
        print("Extrahiere Text-Layer...\n")
        set_phase("Text-Layer extrahieren")
        raw_text = extract_text_layer(pdf_in)

        # 2. Formatiere Text-Layer
        print("Bereinige Text-Layer...\n")
        set_phase("Text bereinigen")
        raw_text = remove_repeated_headers_auto(raw_text, min_count=5, max_header_lines=5)
        raw_text = remove_repeated_footers_auto(raw_text, min_count=5, max_footer_lines=5)
        raw_text = remove_multiple_blank_lines_per_page(raw_text)
        raw_text = remove_consecutive_duplicate_lines(raw_text)
        
        # 3. Bilder extrahieren
        print("Extrahiere Bilder...\n")
        set_phase("Bilder")
        imgs, img_placeholders = extract_images(pdf_in)

        # 3a. Ordner öffnen für manuelle Bildauswahl
        if imgs:
            print(f"\n{len(imgs)} Bilder wurden in den Ordner 'images/' extrahiert.")
            print("Öffne Ordner 'images/' im Finder...")
            try:
                if open_file_or_folder("images/"):
                    print("Ordner wurde geöffnet.")
            except Exception as e:
                print(f"Konnte Ordner nicht öffnen: {e}")
            print("Überprüfe jetzt die Bilder und lösche unerwünschte Dateien aus dem 'images/' Ordner.")
            print("\nDrücke Enter, um fortzufahren, sobald du fertig bist...")
            input()
            closed = close_folder_window("images")
            if closed > 0:
                print("images/ Ordner-Fenster geschlossen.")
            else:
                print("Konnte images/ Finder-Fenster nicht automatisch schließen (evtl. bereits zu).")
            # Aktualisierte Liste nach manueller Bearbeitung
            imgs = [img for img in imgs if os.path.exists(img)]
            print(f"{len(imgs)} Bilder werden an das VLM gesendet.\n")
        else:
            print("Keine Bilder gefunden.\n")

        # 4. Platzhalter einfügen (nur für vorhandene Bilder)
        print("Füge Platzhalter für Bilder ein...\n")
        set_phase("Platzhalter")
        text_with_place = insert_placeholders(raw_text, img_placeholders, imgs)

        # 4a. Erste Laufzeitschätzung basierend auf Historie
        char_est = len(text_with_place)
        image_count = len(imgs)
        estimated_duration = estimate_duration_from_history(stats_history, char_est, image_count)
        acc_percent = compute_eta_accuracy(stats_history, window=6)
        if acc_percent is not None:
            print(f"ETA-Genauigkeit (letzte 6 Läufe): ~{acc_percent}%")
        if estimated_duration:
            est_total = clamp_eta(estimated_duration)
            est_total_in_use = est_total
            set_estimated_total(est_total)
            print(f"Geschätzte Gesamtdauer: ~{int(est_total)}s (Zeichen: {char_est}, Bilder: {image_count}).\n")
        else:
            print("Keine belastbare Laufzeitschätzung möglich (zu wenig Daten).\n")

        # 5. Semantische Bildbeschreibung
        print("Beschreibe Bilder mit MLX-VLM...\n")
        processing_start_ts = time.time()
        set_start_time(processing_start_ts)
        if imgs:
            set_phase("Bilder")
            caps = caption_images(imgs, set_phase_cb=set_phase)
        else:
            caps = {}

        # Zwischen-Update der ETA nach Bildbeschreibung (falls Schätzung vorhanden)
        if estimated_duration:
            elapsed = time.time() - (processing_start_ts or script_start_ts)
            est_total = clamp_eta(max(estimated_duration, elapsed + ETA_MIN_SECONDS))
            est_total_in_use = est_total
            set_estimated_total(est_total)

        # 6. Zusammenführen
        print("Füge Text und Bildbeschreibungen zusammen...\n")
        set_phase("Text")
        final = merge_text(text_with_place, caps)

        # Debug / Sicherheitskopie vor LLM-Formatierung
        raw_out = f"outcome_raw_{now.strftime('%d.%m.%y_%H:%M')}.txt"
        try:
            with open(raw_out, "w") as f:
                f.write(final)
            print(f"Rohtext vor LLM-Formatierung in '{raw_out}' gespeichert (Länge: {len(final)} Zeichen).\n")
        except Exception as e:
            print(f"Konnte Rohtext nicht speichern: {e}")

        # 7. Finalen Text durch ein LLM formatieren lassen
        print("Optimiere die Formatierung des finalen Texts mit MLX-LLM...\n")
        set_phase("Formatieren")
        final = format_ocr(final)
        print(f"Formatierte Textlänge: {len(final)} Zeichen.\n")

        # 8. Ausgabe
        print(f"Schreibe angereicherten Text in '{out_txt}'...\n")
        set_phase("Ergebnisdatei")
        with open(out_txt, "w") as f:
            f.write(final)

        # 9. Bild-Ordner löschen
        print("Bereinige temporäre Dateien...\n")
        set_phase("Aufräumen")
        shutil.rmtree("images", ignore_errors=True)

        # 10. Ergebnisdatei öffnen (optional)
        print(f"Öffne '{out_txt}' automatisch...")
        try:
            open_file_or_folder(out_txt)
        except Exception as e:
            print(f"⚠ Konnte '{out_txt}' nicht öffnen: {e}")

        print(f"Fertig! Datei '{out_txt}' enthält den angereicherten Text.")
        set_phase("Fertig")
        duration = time.time() - (processing_start_ts or script_start_ts)
        delta_sec = round(duration - est_total_in_use, 2) if est_total_in_use is not None else None
        try:
            append_stat(
                {
                    "chars": len(final),
                    "images": len(imgs),
                    "duration_sec": round(duration, 2),
                    "estimated_sec": est_total_in_use if est_total_in_use is not None else "",
                    "delta_sec": delta_sec if delta_sec is not None else "",
                }
            )
            print(f"Statistik aktualisiert in '{STATS_FILE}'.")
        except Exception as e:
            print(f"Warnung: Statistik konnte nicht gespeichert werden ({e}).")
    finally:
        if (
            platform.system() == "Darwin"
            and low_power_changed
            and initial_low_power is not None
            and (args.restore_low_power or args.low_power == "auto")
        ):
            print("Stelle ursprünglichen Batteriesparmodus wieder her...")
            if set_low_power_mode_state(initial_low_power):
                print("Ursprünglicher Batteriesparmodus wiederhergestellt.\n")
            else:
                print("Konnte ursprünglichen Batteriesparmodus nicht wiederherstellen.\n")
        stop_status()
        stop_xbar_if_started(was_running)

if __name__ == "__main__":
    main()
