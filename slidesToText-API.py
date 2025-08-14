#!/usr/bin/env python3
import sys
import os
import fitz
from PIL import Image
import pytesseract, cv2, numpy as np
import shutil
import datetime
import hashlib
from collections import Counter, defaultdict
import re

# Neu: Google AI Studio / Gemini
import google.generativeai as genai

# ---------- PDF/Text Hilfsfunktionen (unverändert zu MLX-Version) ----------
def extract_text_layer(pdf_path):
    doc = fitz.open(pdf_path)
    return [page.get_text() for page in doc]

def image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def extract_images(pdf_path, img_folder="images"):
    os.makedirs(img_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    img_files = []
    img_placeholders = defaultdict(list)
    hash_to_paths = defaultdict(list)

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
        if len(paths) == 1:
            path, page_num = paths[0]
            img_files.append(path)
            img_placeholders[page_num].append(f"[IMG:{os.path.basename(path)}]")
        else:
            for p, _ in paths:
                print(f"Doppeltes Bild erkannt und entfernt: {os.path.basename(p)}")
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
    return img_files, img_placeholders

def insert_placeholders(text_layer_list, img_placeholders):
    result = []
    for i, text in enumerate(text_layer_list):
        placeholders = "\n".join(img_placeholders.get(i, []))
        result.append(text + ("\n" + placeholders if placeholders else ""))
    return "\n\n".join(result)

def merge_text(text_with_place, captions):
    enriched = text_with_place
    for img in captions:
        placeholder = f"[IMG:{os.path.basename(img)}]"
        enriched = enriched.replace(placeholder, f"BILD: [{captions[img]}]\n")
    return enriched

def mask_footer_line(line):
    line = re.sub(r'\b\d+\s*/\s*\d+\b', '', line)
    line = re.sub(r'\bSeite\s*\d+\b', '', line, flags=re.IGNORECASE)
    line = re.sub(r'\d+', '', line)
    return line.strip()

def remove_repeated_headers_auto(text_layer_list, min_count=5, max_header_lines=5):
    print("Suche nach wiederholten Kopfzeilen ...")
    candidates = []
    for n in range(1, max_header_lines+1):
        blocks = ["\n".join(page.splitlines()[:n])
                  for page in text_layer_list if len(page.splitlines()) >= n]
        block_counts = Counter(blocks)
        for block, count in block_counts.items():
            if count >= min_count:
                print(f"Kandidat Kopfzeile ({n} Zeilen, {count}x):\n---\n{block}\n---\n")
                candidates.append((n, block, count))
    if not candidates:
        print("Keine Kopfzeile erkannt.")
        return text_layer_list
    candidates.sort(key=lambda x: (x[0], x[2]), reverse=True)
    header_lines, header_block, _ = candidates[0]
    print(f"Entferne Kopfzeile ({header_lines} Zeilen):\n---\n{header_block}\n---\n")
    cleaned, removed_count, first_found = [], 0, False
    for page in text_layer_list:
        lines = page.splitlines()
        block = "\n".join(lines[:header_lines])
        if block == header_block:
            if not first_found:
                first_found = True
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
                print(f"Kandidat Fußzeile ({n} Zeilen, {count}x):\n---\n{block}\n---\n")
                candidates.append((n, block, count))
    if not candidates:
        print("Keine Fußzeile erkannt.")
        return text_layer_list
    candidates.sort(key=lambda x: (x[0], x[2]), reverse=True)
    footer_lines, footer_block, _ = candidates[0]
    print(f"Entferne Fußzeile ({footer_lines} Zeilen):\n---\n{footer_block}\n---\n")
    cleaned, removed_count, first_found = [], 0, False
    for page in text_layer_list:
        lines = page.splitlines()
        block = "\n".join(mask_footer_line(line) for line in lines[-footer_lines:])
        if block == footer_block:
            if not first_found:
                first_found = True
            else:
                lines = lines[:-footer_lines]
                removed_count += 1
        cleaned.append("\n".join(lines))
    print(f"Fußzeile auf {removed_count} Seiten entfernt.\n")
    return cleaned

def remove_multiple_blank_lines_per_page(text_layer_list):
    cleaned_pages = []
    for page in text_layer_list:
        cleaned = re.sub(r'\n\s*\n+', '\n', page)
        cleaned_pages.append(cleaned.strip())
    return cleaned_pages

def remove_consecutive_duplicate_lines(text_layer_list):
    cleaned_pages = []
    for page in text_layer_list:
        new_lines, prev = [], None
        for line in page.splitlines():
            current = line.strip()
            if current == prev and current != "":
                continue
            new_lines.append(line)
            prev = current
        cleaned_pages.append("\n".join(new_lines))
    return cleaned_pages

# ---------- Neu: Gemini-basierte Funktionen ----------
def _configure_gemini():
    api_key = os.environ.get("***REMOVED***")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY ist nicht gesetzt.")
    genai.configure(api_key=api_key)

def caption_images_gemini(img_files, model_name="gemini-1.5-flash"):
    # Bildbeschreibung über Google AI Studio (Gemini). Ein Request pro Bild:
    # Content = [PIL.Image, Prompt].

    _configure_gemini()
    model = genai.GenerativeModel(model_name)
    print(f"{len(img_files)} Bilder werden mit Gemini beschrieben...\n")
    captions = {}
    prompt = (
        "Beschreibe dieses Bild auf Deutsch. "
        "Bei Fotografie/Szene: max. zwei knappe Sätze. "
        "Bei Diagramm/Skizze/Schemata: sehr genau beschreiben und interpretieren. "
        "Wenn nur Text: gib den Text wörtlich wieder. "
        "Wenn Teile unleserlich sind, weise darauf hin."
    )
    for idx, img_path in enumerate(img_files, 1):
        pil_img = Image.open(img_path).convert("RGB")
        try:
            resp = model.generate_content([pil_img, prompt], request_options={"timeout": 90})
            text = (resp.text or "").strip()
        except Exception as e:
            text = f"(Fehler bei der Bildbeschreibung: {e})"
        captions[img_path] = text
        print(f"Bild {idx}/{len(img_files)} beschrieben: {os.path.basename(img_path)}")
    return captions

def format_ocr_gemini(text, model_name="gemini-1.5-flash"):
    """
    Optionale Textnachformatierung wie in deiner V1 – jetzt via Gemini.
    """
    _configure_gemini()
    model = genai.GenerativeModel(model_name)
    system = (
        "Du bist ein Hilfsprogramm zur Textaufbereitung. "
        "Behalte den Inhalt, aber: entferne Zeilenumbrüche mitten im Satz, "
        "korrigiere Leerzeichen vor Satzzeichen, entferne doppelte/unnötige Leerzeilen, "
        "strukturiere in sinnvolle Absätze. Ändere keine inhaltlichen Aussagen."
    )
    prompt = f"{system}\n\n---\n{text.strip()}\n---\n"
    try:
        resp = model.generate_content(prompt, request_options={"timeout": 120})
        return (resp.text or "").strip()
    except Exception as e:
        return f"(Formatierungsfehler: {e})\n\n{text}"

# ---------- main ----------
def main():
    if len(sys.argv) != 2:
        print("Benutze: pdf_to_text_gemini.py input.pdf")
        sys.exit(1)
    pdf_in  = sys.argv[1]

    now = datetime.datetime.now()
    out_txt = f"outcome_{now.strftime('%d.%m.%y_%H:%M')}.txt"

    print("Extrahiere Text-Layer...\n")
    raw_text = extract_text_layer(pdf_in)

    print("Bereinige Text-Layer...\n")
    raw_text = remove_repeated_headers_auto(raw_text, min_count=5, max_header_lines=5)
    raw_text = remove_repeated_footers_auto(raw_text, min_count=5, max_footer_lines=5)
    raw_text = remove_multiple_blank_lines_per_page(raw_text)
    raw_text = remove_consecutive_duplicate_lines(raw_text)

    print("Extrahiere Bilder...\n")
    imgs, img_placeholders = extract_images(pdf_in)

    print("Füge Platzhalter für Bilder ein...\n")
    text_with_place = insert_placeholders(raw_text, img_placeholders)

    print("Beschreibe Bilder mit Gemini...\n")
    caps = caption_images_gemini(imgs, model_name="gemini-1.5-flash")

    print("Füge Text und Bildbeschreibungen zusammen...\n")
    final = merge_text(text_with_place, caps)

    # Optional: Nachformatierung (kannst du auch auskommentieren)
    # print("Optimiere die Formatierung mit Gemini...\n")
    # final = format_ocr_gemini(final, model_name="gemini-1.5-flash")

    print(f"Schreibe angereicherten Text in '{out_txt}'...\n")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(final)

    print("Bereinige temporäre Dateien...\n")
    shutil.rmtree("images", ignore_errors=True)
    print(f"Fertig! Datei '{out_txt}' enthält den angereicherten Text.")

if __name__ == "__main__":
    main()
