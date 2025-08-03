#!/usr/bin/env python3
import sys
import os
import fitz                         # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import easyocr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

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

def ocr_on_images(img_files, lang="deu"):
    # Tesseract via pytesseract
    ocr_texts = {}
    for img in img_files:
        txt = pytesseract.image_to_string(img, lang=lang)
        ocr_texts[img] = txt.strip()
    return ocr_texts

def caption_images(img_files, device="cpu"):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    captions = {}
    for img in img_files:
        image = Image.open(img).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        out    = model.generate(**inputs)
        cap    = processor.decode(out[0], skip_special_tokens=True)
        captions[img] = cap
    return captions

def merge_text(text_layer, ocr_texts, captions):
    # Setze Platzhalter für jedes Bild
    enriched = text_layer
    for img in ocr_texts:
        placeholder = f"[IMG:{os.path.basename(img)}]"
        # Falls OCR Text vorhanden, hänge ihn an die Caption an
        combined = captions[img]
        if ocr_texts[img]:
            combined += "\n(Erkannter Text im Bild: " + ocr_texts[img] + ")"
        enriched = enriched.replace(placeholder, combined)
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
        print("Usage: pdf_to_text.py input.pdf output.txt")
        sys.exit(1)
    pdf_in  = sys.argv[1]
    out_txt  = sys.argv[2]

    # 1. Text-Layer extrahieren
    raw_text = extract_text_layer(pdf_in)

    # 2. Bilder extrahieren
    imgs = extract_images(pdf_in)

    # 3. Platzhalter einfügen
    text_with_place = insert_placeholders(raw_text, imgs)

    # 4. OCR auf Bildern
    ocrs = ocr_on_images(imgs)

    # 5. Semantische Bildbeschreibung
    caps = caption_images(imgs)

    # 6. Zusammenführen
    final = merge_text(text_with_place, ocrs, caps)

    # 7. Ausgabe
    with open(out_txt, "w") as f:
        f.write(final)

    print(f"Fertig! Datei '{out_txt}' enthält den angereicherten Text.")

if __name__ == "__main__":
    main()