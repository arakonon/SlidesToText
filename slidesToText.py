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


# Optional: für ChatGPT-Zusammenfassung
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_layer(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []
    for page in doc:
        txt = page.get_text()
        full_text.append(txt)
    return "\n".join(full_text)

def extract_images(pdf_path, img_folder="images"):
    os.makedirs(img_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    img_files = []
    for page in doc:
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            data = base["image"]
            ext = base["ext"]
            fname = f"page{page.number+1}_{img_index}.{ext}"
            path = os.path.join(img_folder, fname)
            with open(path, "wb") as f:
                f.write(data)
            img_files.append(path)
    return img_files

def caption_images(img_files,
                   model_path="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"):
    # Modell einmalig laden
    model, processor = load(model_path)
    cfg = load_config(model_path)

    captions = {}
    for img in img_files:
        # MLX-VLM erwartet eine Liste von Bildern
        pil_img = [Image.open(img).convert("RGB")]

        # Einfache Prompt-Vorlage
        prompt = "Beschreibe dieses Bild sehr genau, auf Deutsch. Wenn es nur Text enthält, gib den Text wieder. "
        prompt_fmt = apply_chat_template(processor, cfg,
                                         prompt, num_images=len(pil_img))

        # Inferenz
        cap = generate(model, processor, prompt_fmt, pil_img,
                       verbose=False)  # returns str
        captions[img] = cap.text.strip()

    return captions

def merge_text(text_layer, captions):
    enriched = text_layer
    for img in captions:
        placeholder = f"[IMG:{os.path.basename(img)}]"
        # Hier caption anpassen
        # z.B. "BILD: ..." oder nur den Text
        enriched = enriched.replace(placeholder, f"BILD: [{captions[img]}]\n\n")
    return enriched

def insert_placeholders(text_layer, img_files):
    # Füge am Ende jeder Seite einen Platzhalter ein
    doc = fitz.open()
    # Wir nutzen hier einfach den ursprünglichen Text, darum Dummy:
    # Du kannst auch die genaue Position im Text suchen und markieren.
    # Hier als einfache Variante: ganz unten pro Seite.
    return text_layer + "\n\n" + "\n".join(f"[IMG:{os.path.basename(i)}]" for i in img_files)

def main():
    if len(sys.argv) != 3:
        print("Benutze: pdf_to_text.py input.pdf output.txt")
        sys.exit(1)
    pdf_in  = sys.argv[1]

    # Erzeuge Dateinamen mit Datum und Uhrzeit
    now = datetime.datetime.now()
    out_txt = f"outcome_{now.strftime('%d.%m.%y_%H:%M')}.txt"

    # 1. Text-Layer extrahieren
    print("Extrahiere Text-Layer...\n")
    raw_text = extract_text_layer(pdf_in)

    # 2. Bilder extrahieren
    print("Extrahiere Bilder...\n")
    imgs = extract_images(pdf_in)

    # 3. Platzhalter einfügen
    print("Füge Platzhalter für Bilder ein...\n")
    text_with_place = insert_placeholders(raw_text, imgs)

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
    print("Bereinige temporäre Dateien...\n")
    shutil.rmtree("images", ignore_errors=True)

    print(f"Fertig! Datei '{out_txt}' enthält den angereicherten Text.")

if __name__ == "__main__":
    main()