#!/usr/bin/env python3
import sys
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

def caption_images(img_files, model_path="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"):
    print(f"{len(img_files)} Bilder werden beschrieben...\n")
    model, processor = load_vlm(model_path)
    cfg = load_config(model_path)
    bildNr = 0
    captions = {}
    for img in img_files:
        pil_img = [Image.open(img).convert("RGB").resize((512, 512))]
        prompt = "Beschreibe dieses Bild auf Deutsch. Wenn es sich um eine Fotografie oder Szene handelt, beschreibe in maximal 2 kurzen Sätzen. Wenn es sich um ein Diagramm, eine Skizze oder eine schematische Darstellung handelt, beschreibe das Bild sehr genau und interpretiere es. Wenn das Bild nur Text enthält, gib nur den Text wieder. Wenn Teile des Bildes nicht erkennbar sind, weise darauf hin."
        prompt_fmt = apply_chat_template(processor, cfg, prompt, num_images=len(pil_img))
        cap = generate_vlm(model, processor, prompt_fmt, pil_img, verbose=False)
        captions[img] = cap.text.strip()
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
    # 4-Bit-Modell laden
    model_format, tok = load_lm(
        "mlx-community/Qwen1.5-1.8B-Chat-4bit",
        tokenizer_config={
            "eos_token": "<|endoftext|>",
            "trust_remote_code": True,
        },
    )
    # Systemnachricht für das LLM
    system = ("Du bist ein Hilfsprogramm zur Textaufbereitung. Behalte den Inhalt, aber: entferne Zeilenumbrüche mitten im Satz, korrigiere Leerzeichen vor Satzzeichen, entferne doppelte oder unnötige Leerzeilen und achte auf sinnvolle Absätze, Ändere keine inhaltlichen Aussagen.\n")
    messages = [
        {system},
        {"content": text.strip()}
    ]
    try:
        # Bevorzugte Chat-Vorlage des Tokenizers nutzen
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
    except (AttributeError, ValueError):
        # Fallback: einfach System + User‑Text hintereinander
        prompt = system + "\n\n" + text.strip() + "\n\n"
    out = generate_lm(model_format, tok, prompt=prompt, max_tokens=len(text)//2)
    return out.strip()

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

    # 2. Formatiere Text-Layer
    print("Bereinige Text-Layer...\n")
    raw_text = remove_repeated_headers_auto(raw_text, min_count=5, max_header_lines=5)
    raw_text = remove_repeated_footers_auto(raw_text, min_count=5, max_footer_lines=5)
    raw_text = remove_multiple_blank_lines_per_page(raw_text)
    raw_text = remove_consecutive_duplicate_lines(raw_text)
    
    # 3. Bilder extrahieren
    print("Extrahiere Bilder...\n")
    imgs, img_placeholders = extract_images(pdf_in)

    # 4. Platzhalter einfügen
    print("Füge Platzhalter für Bilder ein...\n")
    text_with_place = insert_placeholders(raw_text, img_placeholders)

    # 5. Semantische Bildbeschreibung
    print("Beschreibe Bilder mit MLX-VLM (Qwen2.5)...\n")
    caps = caption_images(imgs)

    # 6. Zusammenführen
    print("Füge Text und Bildbeschreibungen zusammen...\n")
    final = merge_text(text_with_place, caps)
    
    # 7. Finalen Text durch ein LLM formatieren lassen
    # print("Optimiere die Formatierung des finalen Texts mit MLX-LLM (Qwen1.5)...\n")
    # final = format_ocr(final)

    # 8. Ausgabe
    print(f"Schreibe angereicherten Text in '{out_txt}'...\n")
    with open(out_txt, "w") as f:
        f.write(final)

    # 9. Bild-Ordner löschen
    print("Bereinige temporäre Dateien...\n")
    shutil.rmtree("images", ignore_errors=True)

    print(f"Fertig! Datei '{out_txt}' enthält den angereicherten Text.")

if __name__ == "__main__":
    main()