#!/usr/bin/env python3
import sys
import os
import fitz                         # PyMuPDF
from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import pytesseract, cv2, numpy as np
from PIL import Image
import shutil
import datetime
import hashlib
from collections import Counter


# Optional: für ChatGPT-Zusammenfassung
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")

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

    # Nur eindeutige Bilder behalten
    for paths in hash_to_paths.values():
        if len(paths) == 1:
            path, page_num = paths[0]
            img_files.append(path)
            img_placeholders[page_num].append(f"[IMG:{os.path.basename(path)}]")
        else:
            for p, _ in paths:
                print(f"Doppeltes Bild erkannt und entfernt: {os.path.basename(p)} \n")
                os.remove(p)
    return img_files, img_placeholders

def caption_images(img_files,
                   model_path="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"):
    print(f"{len(img_files)} Bilder werden beschrieben...\n")  # <--- Hier die Ausgabe
    bildNr = 0
    captions = {}
    for img in img_files:
        model, processor = load(model_path)
        cfg = load_config(model_path)
        pil_img = [Image.open(img).convert("RGB").resize((512, 512))]
        prompt = "Beschreibe dieses Bild auf Deutsch. Wenn es sich um eine Fotografie oder Szene handelt, beschreibe in maximal 2 kurzen Sätzen. Wenn es sich um ein Diagramm, eine Skizze oder eine schematische Darstellung handelt, beschreibe das Bild sehr genau und interpretiere es. Wenn das Bild nur Text enthält, gib nur den Text wieder. Wenn Teile des Bildes nicht erkennbar sind, weise darauf hin."
        prompt_fmt = apply_chat_template(processor, cfg, prompt, num_images=len(pil_img))
        cap = generate(model, processor, prompt_fmt, pil_img, verbose=False)
        captions[img] = cap.text.strip()
        del model, processor, cfg, pil_img, cap  # Speicher freigeben
        bildNr += 1
        print("\nBild Nr.", bildNr, "beschrieben\n")
    return captions

def merge_text(text_with_place, captions):
    enriched = text_with_place
    for img in captions:
        placeholder = f"[IMG:{os.path.basename(img)}]"
        enriched = enriched.replace(placeholder, f"BILD: [{captions[img]}] \n")
    return enriched

def insert_placeholders(text_layer_list, img_placeholders):
    # text_layer_list: Liste mit Text pro Seite
    # img_placeholders: Dict {seitennummer: [platzhalter, ...]}
    result = []
    for i, text in enumerate(text_layer_list):
        placeholders = "\n".join(img_placeholders.get(i, []))
        result.append(text + ("\n" + placeholders if placeholders else ""))
    return "\n\n".join(result)

def image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def remove_repeated_headers(text_layer_list, min_count=5):
    # Zerlege jede Seite in Zeilen
    first_lines = [page.splitlines()[0] for page in text_layer_list if page.strip()]
    # Zähle, welche Zeilen oft vorkommen
    line_counts = Counter(first_lines)
    # Finde die Zeilen, die mehrfach vorkommen
    repeated = {line for line, count in line_counts.items() if count >= min_count}
    # Entferne diese Zeilen am Seitenanfang
    cleaned = []
    for page in text_layer_list:
        lines = page.splitlines()
        if lines and lines[0] in repeated:
            lines = lines[1:]
        cleaned.append("\n".join(lines))
    return cleaned

def main():
    if len(sys.argv) != 2:
        print("Benutze: pdf_to_text.py input.pdf")
        sys.exit(1)
    pdf_in  = sys.argv[1]

    # Erzeuge Dateinamen mit Datum und Uhrzeit
    now = datetime.datetime.now()
    out_txt = f"outcome_{now.strftime('%d.%m.%y_%H:%M')}.txt"

    # 1. Text-Layer extrahieren
    print("Extrahiere Text-Layer...\n")
    raw_text = extract_text_layer(pdf_in)
    raw_text = remove_repeated_headers(raw_text, min_count=3)  # min_count ggf. anpassen

    # 2. Bilder extrahieren
    print("Extrahiere Bilder...\n")
    imgs, img_placeholders = extract_images(pdf_in)

    # 3. Platzhalter einfügen
    print("Füge Platzhalter für Bilder ein...\n")
    text_with_place = insert_placeholders(raw_text, img_placeholders)

    # 4. Semantische Bildbeschreibung
    print("Beschreibe Bilder mit MLX-VLM...\n")
    caps = caption_images(imgs)

    # 5. Zusammenführen
    print("Füge Text und Bildbeschreibungen zusammen...\n")
    final = merge_text(text_with_place, caps)

    # 6. Ausgabe
    print(f"Schreibe angereicherten Text in '{out_txt}'...\n")
    with open(out_txt, "w") as f:
        f.write(final)

    # 7. Bild-Ordner löschen
    #print("Bereinige temporäre Dateien...\n")
    #shutil.rmtree("images", ignore_errors=True)

    print(f"Fertig! Datei '{out_txt}' enthält den angereicherten Text.")

if __name__ == "__main__":
    main()