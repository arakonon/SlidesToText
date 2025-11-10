#!/usr/bin/env python3
import sys
import os
import fitz  # PyMuPDF
from PIL import Image
from mlx_vlm import load as load_vlm, generate as generate_vlm
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import hashlib
import datetime
import subprocess
import platform
import shutil
import glob  # für move_old_outcome_files

def image_hash(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def extract_images(pdf_path: str, img_folder: str = "images"):
    """
    Extrahiert Bilder, dedupliziert per SHA256 und gibt die Liste der eindeutigen Bildpfade zurück.
    Doppelte Bilder werden gelöscht.
    """
    os.makedirs(img_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    hash_to_paths = {}
    kept = []
    removed = 0

    for page_num, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base = doc.extract_image(xref)
                data = base["image"]
                ext = base.get("ext", "png")
                fname = f"page{page_num}_{img_index}.{ext}"
                path = os.path.join(img_folder, fname)
                with open(path, "wb") as f:
                    f.write(data)
                h = image_hash(path)
                if h in hash_to_paths:
                    # Duplikat: Datei entfernen
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                    removed += 1
                else:
                    hash_to_paths[h] = path
            except Exception as e:
                print(f"Warnung: Konnte ein Bild nicht extrahieren (Seite {page_num}, Index {img_index}): {e}")

    kept = list(hash_to_paths.values())
    if removed:
        print(f"{removed} doppelte Bild(er) entfernt.")
    print(f"{len(kept)} eindeutige Bild(er) extrahiert.")
    return kept

def open_file_or_folder(path: str) -> bool:
    try:
        if platform.system() == "Darwin":
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

def close_folder_window(folder_path: str) -> int:
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

def caption_images(img_files, model_path="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"):
    """
    Erzeugt deutschsprachige Bildbeschreibungen mit MLX-VLM.
    Rückgabe: Liste von (basename, caption)
    """
    if not img_files:
        return []

    print(f"{len(img_files)} Bild(er) werden beschrieben...")
    model, processor = load_vlm(model_path)
    cfg = load_config(model_path)

    prompt = (
        "Beschreibe dieses Bild auf Deutsch. "
        "Wenn es sich um eine Fotografie oder Szene handelt, beschreibe in maximal 2 kurzen Sätzen. "
        "Wenn es sich um ein Diagramm, eine Skizze oder eine schematische Darstellung handelt, "
        "beschreibe das Bild sehr genau und interpretiere es. "
        "Wenn das Bild nur Text enthält, gib nur den Text wieder. "
        "Wenn Teile des Bildes nicht erkennbar sind, weise darauf hin."
    )

    results = []
    for idx, img in enumerate(img_files, 1):
        pil_img = [Image.open(img).convert("RGB").resize((512, 512))]
        prompt_fmt = apply_chat_template(processor, cfg, prompt, num_images=len(pil_img))
        cap = generate_vlm(model, processor, prompt_fmt, pil_img, verbose=False)
        text = (cap.text or "").strip()
        print(f"Bild {idx}/{len(img_files)} beschrieben: {os.path.basename(img)}")
        results.append((os.path.basename(img), text))
    return results

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

def main():
    if len(sys.argv) < 2:
        print("Benutzung: pictureDescription-MLX.py input.pdf [--keep-images] [--no-open]")
        sys.exit(1)
    # Alte outcome-Dateien archivieren
    move_old_outcome_files()

    pdf_in = sys.argv[1]
    keep_images = "--keep-images" in sys.argv[2:]
    no_open = "--no-open" in sys.argv[2:]

    if not os.path.isfile(pdf_in):
        print(f"Datei nicht gefunden: {pdf_in}")
        sys.exit(1)

    img_dir = "images"
    print("Extrahiere Bilder...")
    imgs = extract_images(pdf_in, img_folder=img_dir)

    if not imgs:
        print("Keine Bilder gefunden.")
        sys.exit(0)

    if not no_open:
        print("Öffne Ordner 'images/' im Finder. Lösche unerwünschte Bilder und kehre zurück.")
        opened = open_file_or_folder(img_dir)
        if not opened:
            print("Hinweis: Konnte den Ordner nicht automatisch öffnen.")
    input("Drücke Enter, sobald du die unerwünschten Bilder gelöscht hast...")

    closed = close_folder_window(img_dir)
    if closed > 0:
        print("images/ Ordner-Fenster geschlossen.")

    # Aktualisierte Liste (nur nicht gelöschte Bilder)
    imgs = [p for p in imgs if os.path.exists(p)]
    if not imgs:
        print("Alle Bilder wurden gelöscht. Nichts zu beschreiben.")
        if not keep_images:
            shutil.rmtree(img_dir, ignore_errors=True)
        sys.exit(0)

    print(f"{len(imgs)} Bild(er) werden an das VLM gesendet...")
    descriptions = caption_images(imgs)

    # Ausgabe
    now = datetime.datetime.now().strftime("%d.%m.%y_%H-%M")
    out_txt = f"pictureDescriptions_{now}.txt"
    with open(out_txt, "w") as f:
        for name, desc in descriptions:
            f.write(f"{name}:\n{desc}\n\n")

    print("\nBeschreibungen (nur nicht gelöschte Bilder):\n")
    for name, desc in descriptions:
        print(f"{name}:\n{desc}\n")

    print(f"Gespeichert in: {out_txt}")

    if not keep_images:
        print("Bereinige temporären Ordner 'images/'...")
        shutil.rmtree(img_dir, ignore_errors=True)

    print(f"Öffne '{out_txt}' automatisch...")
    try:
        open_file_or_folder(out_txt)
    except Exception as e:
        print(f"⚠ Konnte '{out_txt}' nicht öffnen: {e}")

if __name__ == "__main__":
    main()