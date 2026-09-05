"""
Módulo de rescate de recuadros, tablas y texto omitido por el clasificador de Docling.
Garantiza que ningún recuadro con fondo de color, tabla o campo fiscal clasificado
erróneamente como 'picture' o ignorado por el layout se pierda en la extracción.
"""
import os
import re
from typing import Any, Dict, List, Optional, Tuple
try:
    import numpy as np
except ImportError:
    np = None
from PIL import Image

from core.geometry import compute_bounding_box, tokenize_text_to_spatial_tokens
from core.sanitizer import fix_split_accents, fix_fiscal_identifiers
try:
    from core.extractors import cluster_horizontal_line_items, match_line_item_pattern
except Exception:
    def cluster_horizontal_line_items(elements, id_counter):
        return elements
    def match_line_item_pattern(text):
        return False

# Singleton para el motor de OCR
_OCR_ENGINE = None

def get_ocr_engine():
    """Obtiene o inicializa el motor de OCR ligero (RapidOCR preferente o EasyOCR/Tesseract)."""
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    # 1. Intentar RapidOCR (ONNX Runtime, ultra-rápido en GPU y CPU)
    try:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_ENGINE = ("rapidocr", RapidOCR())
        print("[Box Recovery] ✅ Motor RapidOCR inicializado correctamente.", flush=True)
        return _OCR_ENGINE
    except Exception as e:
        print(f"[Box Recovery] ⚠️ RapidOCR no disponible ({e}), probando alternativas...", flush=True)

    # 2. Intentar EasyOCR
    try:
        import easyocr
        reader = easyocr.Reader(["es", "en"], gpu=False)
        _OCR_ENGINE = ("easyocr", reader)
        print("[Box Recovery] ✅ Motor EasyOCR inicializado como fallback.", flush=True)
        return _OCR_ENGINE
    except Exception as e:
        print(f"[Box Recovery] ⚠️ EasyOCR no disponible ({e}), probando Tesseract...", flush=True)

    # 3. Fallback Tesseract CLI
    _OCR_ENGINE = ("tesseract", None)
    return _OCR_ENGINE


import unicodedata

def normalize_text_for_comparison(text: str) -> str:
    """Normaliza texto eliminando acentos, diacríticos y homogeneizando 0/O."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    return stripped.lower().replace("0", "o")


def clean_token_word(w: str) -> str:
    """Limpia puntuación y tildes para comparación insensible de tokens."""
    w_norm = normalize_text_for_comparison(w)
    return re.sub(r'\W+', '', w_norm)


def is_text_already_in_markdown(text: str, markdown: str) -> bool:
    """Comprueba si una porción significativa del texto ya está presente en el markdown."""
    if not text or not markdown:
        return False
    clean_t = clean_token_word(text)
    clean_m = clean_token_word(markdown)
    if len(clean_t) < 5:
        return clean_t in clean_m
    # Buscar subsecuencias de 15 caracteres para mayor sensibilidad en datos cortos (fechas, DNI, importes)
    if len(clean_t) < 15:
        return clean_t in clean_m
    return (clean_t[:15] in clean_m) or (clean_t[-15:] in clean_m)


def refine_tokens_with_ocr_line(
    line_text: str,
    line_bbox: Dict[str, float],
    page_tokens: List[Dict[str, Any]]
) -> bool:
    """
    Refina la geometría de tokens existentes utilizando la detección exacta de línea de RapidOCR/Tesseract.
    Evita que tokens en párrafos multilínea queden deformados verticalmente o desplazados horizontalmente.
    Soporta coincidencia por lista de palabras o por texto concatenado (cuando el OCR omite espacios).
    """
    line_words = [clean_token_word(w) for w in line_text.split() if clean_token_word(w)]
    if not line_words:
        return False

    n = len(line_words)
    w = max(abs(line_bbox["x2"] - line_bbox["x1"]), 1.0)
    x1 = line_bbox["x1"]
    y1 = line_bbox["y1"]
    y2 = line_bbox["y2"]

    # 1. Búsqueda por secuencia de palabras separadas
    best_match_indices: Optional[Tuple[int, int]] = None
    min_y_dist = float("inf")

    for i in range(len(page_tokens) - n + 1):
        match = True
        for j in range(n):
            if clean_token_word(page_tokens[i + j].get("text", "")) != line_words[j]:
                match = False
                break

        if match:
            tok0_bb = page_tokens[i].get("bbox")
            y_dist = abs(tok0_bb["y1"] - y1) if (tok0_bb and "y1" in tok0_bb) else 0.0
            # Si hay varios matches en la página (ej. "INSTALACIONES Y OBRAS DE GALICIA, S.L."),
            # elegir el que esté espacialmente más cerca de la detección OCR actual
            if y_dist < min_y_dist:
                min_y_dist = y_dist
                best_match_indices = (i, i + n)

    # 2. Búsqueda por concatenación continua si no hubo coincidencia por palabras separadas
    # (ej: RapidOCR o Tesseract detecta 'PADREDONRUA14,BAJO' o '36203VIG0-PONTEVEDRA')
    if not best_match_indices:
        line_concat = "".join(line_words)
        for i in range(len(page_tokens)):
            accum = ""
            for j in range(i, min(len(page_tokens), i + 12)):
                accum += clean_token_word(page_tokens[j].get("text", ""))
                if accum == line_concat:
                    tok0_bb = page_tokens[i].get("bbox")
                    y_dist = abs(tok0_bb["y1"] - y1) if (tok0_bb and "y1" in tok0_bb) else 0.0
                    if y_dist < min_y_dist:
                        min_y_dist = y_dist
                        best_match_indices = (i, j + 1)
                    break
                if len(accum) > len(line_concat):
                    break

    if best_match_indices and min_y_dist < 400.0:
        start_idx, end_idx = best_match_indices
        matched_tokens = page_tokens[start_idx:end_idx]
        total_tok_chars = max(sum(len(t.get("text", "")) for t in matched_tokens), 1)

        # Distribuir las coordenadas x1..x2 proporcionalmente al tamaño de cada token en la línea
        curr_x = x1
        for tok in matched_tokens:
            t_str = tok.get("text", "")
            t_w = (len(t_str) / total_tok_chars) * w
            tok_x1 = round(curr_x, 2)
            tok_x2 = round(curr_x + t_w, 2)
            curr_x += t_w

            tok["bbox"] = {
                "x1": min(tok_x1, tok_x2),
                "y1": y1,
                "x2": max(tok_x1, tok_x2),
                "y2": y2
            }
        return True

    return False


def render_page_image(source_path: str, page_num: int = 1) -> Optional[Image.Image]:
    """Renderiza la página del documento a imagen PIL a 300 DPI."""
    if not os.path.exists(source_path):
        return None

    ext = os.path.splitext(source_path)[1].lower()
    if ext != ".pdf":
        try:
            return Image.open(source_path).convert("RGB")
        except Exception as e:
            print(f"[Box Recovery] ⚠️ Error abriendo imagen {source_path}: {e}", flush=True)
            return None

    # Renderizar PDF con pypdfium2
    try:
        import pypdfium2 as pdfium
        pdf_doc = pdfium.PdfDocument(source_path)
        idx = max(0, page_num - 1)
        if idx < len(pdf_doc):
            page = pdf_doc[idx]
            # scale=4.166667 equivale a 300 DPI (72 * 4.166667 = 300)
            bitmap = page.render(scale=4.166667)
            return bitmap.to_pil().convert("RGB")
    except Exception as e:
        print(f"[Box Recovery] ⚠️ Error renderizando PDF con pypdfium2: {e}", flush=True)

    # Fallback con pdftoppm si pypdfium2 falla
    try:
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
            tmp_base = tmp_out.name.replace(".png", "")
        cmd = ["pdftoppm", "-png", "-r", "300", "-f", str(page_num), "-l", str(page_num), source_path, tmp_base]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rendered_file = f"{tmp_base}-{page_num}.png"
        if not os.path.exists(rendered_file):
            rendered_file = f"{tmp_base}-1.png"
        if os.path.exists(rendered_file):
            img = Image.open(rendered_file).convert("RGB")
            try:
                os.remove(rendered_file)
            except Exception:
                pass
            return img
    except Exception as e:
        print(f"[Box Recovery] ⚠️ Error en renderizado con pdftoppm: {e}", flush=True)

    return None


def run_ocr_on_image(pil_img: Image.Image) -> List[Tuple[List[List[float]], str, float]]:
    """
    Ejecuta el motor OCR configurado sobre una imagen PIL.
    Devuelve lista de (box_points_4_corners, text, confidence).
    """
    engine_type, engine_obj = get_ocr_engine()
    results: List[Tuple[List[List[float]], str, float]] = []

    if engine_type == "rapidocr" and engine_obj:
        try:
            np_img = np.array(pil_img)
            ocr_res, _ = engine_obj(np_img)
            if ocr_res:
                for item in ocr_res:
                    if item and len(item) >= 3:
                        box, txt, conf = item[0], item[1], item[2]
                        if txt and str(txt).strip():
                            results.append((box, str(txt).strip(), float(conf or 0.95)))
        except Exception as e:
            print(f"[Box Recovery] ⚠️ Error en RapidOCR: {e}", flush=True)

    elif engine_type == "easyocr" and engine_obj:
        try:
            np_img = np.array(pil_img)
            ocr_res = engine_obj.readtext(np_img)
            if ocr_res:
                for item in ocr_res:
                    if len(item) >= 3:
                        box, txt, conf = item[0], item[1], item[2]
                        if txt and str(txt).strip():
                            results.append((box, str(txt).strip(), float(conf or 0.95)))
        except Exception as e:
            print(f"[Box Recovery] ⚠️ Error en EasyOCR: {e}", flush=True)

    else:
        # Fallback Tesseract TSV
        try:
            import subprocess
            import tempfile
            dpi = pil_img.info.get("dpi") or (300, 300)
            dpi_val = (int(round(dpi[0])), int(round(dpi[1]))) if isinstance(dpi, tuple) else (300, 300)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:
                pil_img.save(tmp_f.name, format="PNG", dpi=dpi_val)
                tmp_path = tmp_f.name
            tsv_out = subprocess.check_output(
                ["tesseract", tmp_path, "stdout", "--psm", "11", "tsv"],
                stderr=subprocess.DEVNULL
            ).decode("utf-8", errors="replace")
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            line_dict = {}
            lines = tsv_out.strip().split("\n")
            for row in lines[1:]:
                parts = row.split("\t")
                if len(parts) >= 12:
                    text = parts[11].strip()
                    conf = float(parts[10] or 0)
                    if text and conf > 30:
                        key = (parts[2], parts[3], parts[4])
                        x = int(parts[6])
                        y = int(parts[7])
                        w = int(parts[8])
                        h = int(parts[9])
                        if key not in line_dict:
                            line_dict[key] = {"words": [], "x1": x, "y1": y, "x2": x + w, "y2": y + h, "conf": conf}
                        line_dict[key]["words"].append(text)
                        line_dict[key]["x1"] = min(line_dict[key]["x1"], x)
                        line_dict[key]["y1"] = min(line_dict[key]["y1"], y)
                        line_dict[key]["x2"] = max(line_dict[key]["x2"], x + w)
                        line_dict[key]["y2"] = max(line_dict[key]["y2"], y + h)

            for data in line_dict.values():
                b_pts = [
                    [data["x1"], data["y1"]],
                    [data["x2"], data["y1"]],
                    [data["x2"], data["y2"]],
                    [data["x1"], data["y2"]]
                ]
                results.append((b_pts, " ".join(data["words"]), data["conf"] / 100.0))
        except Exception as e:
            print(f"[Box Recovery] ⚠️ Error en Tesseract fallback: {e}", flush=True)

    return results


def recover_missing_boxes_and_content(
    sorted_pages: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
    clean_md: str,
    source_path: str,
    id_counter: Dict[str, int]
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Punto de entrada principal para rescatar recuadros, tablas y texto omitido por Docling.
    Compara las detecciones de alta resolución con el markdown existente, añade las líneas
    faltantes a `tokens`, a `sorted_pages['elements']` e inyecta el contenido en `clean_md`.
    """
    if not source_path or not os.path.exists(source_path):
        return clean_md, sorted_pages, tokens

    print("[Box Recovery] 🔍 Iniciando inspección de recuadros y contenido omitido...", flush=True)

    total_recovered_count = 0
    recovered_md_blocks: List[str] = []

    for p_idx, page_obj in enumerate(sorted_pages):
        page_num = page_obj.get("page", p_idx + 1)
        page_w = float(page_obj.get("width") or 595.2)
        page_h = float(page_obj.get("height") or 841.92)

        # 1. Obtener imagen renderizada de la página
        pil_page = render_page_image(source_path, page_num)
        if not pil_page:
            continue

        img_w, img_h = pil_page.size
        scale_x = img_w / page_w
        scale_y = img_h / page_h

        # 2. Ejecutar OCR completo sobre la página
        ocr_detections = run_ocr_on_image(pil_page)
        if not ocr_detections:
            continue

        current_page_tokens = [t for t in tokens if t.get("page") == page_num]
        recovered_elements: List[Dict[str, Any]] = []

        for box_pts, raw_text, conf in ocr_detections:
            line_text = fix_fiscal_identifiers(fix_split_accents(raw_text.strip()))
            if not line_text or len(line_text) < 2:
                continue

            # Convertir coordenadas de píxeles (Top-Left 0,0) a dimensiones de página (Top-Left 0,0)
            min_px_x = min(p[0] for p in box_pts)
            max_px_x = max(p[0] for p in box_pts)
            min_px_y = min(p[1] for p in box_pts)
            max_px_y = max(p[1] for p in box_pts)

            pt_x1 = round(min_px_x / scale_x, 2)
            pt_x2 = round(max_px_x / scale_x, 2)
            pt_y1 = round(min_px_y / scale_y, 2)
            pt_y2 = round(max_px_y / scale_y, 2)

            bbox = {
                "x1": min(pt_x1, pt_x2),
                "y1": min(pt_y1, pt_y2),
                "x2": max(pt_x1, pt_x2),
                "y2": max(pt_y1, pt_y2)
            }

            # 2.1 Refinar geometría de tokens existentes con la detección exacta de OCR de alta resolución
            if refine_tokens_with_ocr_line(line_text, bbox, current_page_tokens):
                continue

            # Si el texto ya está presente en el Markdown de Docling, no duplicarlo
            if is_text_already_in_markdown(line_text, clean_md):
                continue

            # 3. Tokenización espacial inmutable
            tok_ids, tok_list = tokenize_text_to_spatial_tokens(line_text, bbox, page_num, id_counter)
            tokens.extend(tok_list)

            # 4. Crear línea y bloque
            id_counter["line"] = id_counter.get("line", 0) + 1
            line_id = f"l_rec_{id_counter['line']}"
            line_obj = {
                "line_id": line_id,
                "text": line_text,
                "bbox": bbox,
                "token_ids": tok_ids
            }

            id_counter["block"] = id_counter.get("block", 0) + 1
            block_id = f"b_rec_{id_counter['block']}"
            element_obj = {
                "block_id": block_id,
                "label": "text",
                "text": line_text,
                "bbox": bbox,
                "lines": [line_obj],
                "token_ids": tok_ids,
                "table_data": None
            }

            recovered_elements.append(element_obj)
            total_recovered_count += 1

        if not recovered_elements:
            continue

        # 5. Agrupar fragmentos horizontales de conceptos + importes en line items
        clustered_recovered = cluster_horizontal_line_items(recovered_elements, id_counter)

        # 6. Insertar en los elementos de la página y ordenar por Y2 descendente (de arriba a abajo)
        if "elements" not in page_obj or not isinstance(page_obj["elements"], list):
            page_obj["elements"] = []

        page_obj["elements"].extend(clustered_recovered)
        page_obj["elements"].sort(
            key=lambda el: el.get("bbox", {}).get("y2", 0) if el.get("bbox") else 0,
            reverse=True
        )

        # 7. Formatear el bloque Markdown rescatado en orden espacial
        clustered_recovered.sort(
            key=lambda el: el.get("bbox", {}).get("y2", 0) if el.get("bbox") else 0,
            reverse=True
        )

        table_lines = []
        regular_lines = []
        for el in clustered_recovered:
            txt = el.get("text", "").strip()
            if not txt:
                continue
            if el.get("label") == "line_item" or "|" in txt:
                table_lines.append(txt)
            else:
                regular_lines.append(txt)

        page_rec_md_parts = []
        if regular_lines:
            page_rec_md_parts.append("\n".join(regular_lines))

        if table_lines:
            md_table_rows = ["\n| CONCEPTO / DESCRIPCIÓN | IMPORTE |", "| :--- | :--- |"]
            for tl in table_lines:
                cells = [c.strip() for c in tl.split("|") if c.strip()]
                if len(cells) >= 2:
                    md_table_rows.append(f"| {' '.join(cells[:-1])} | {cells[-1]} |")
                else:
                    md_table_rows.append(f"| {tl} | |")
            page_rec_md_parts.append("\n".join(md_table_rows))

        if page_rec_md_parts:
            recovered_md_blocks.append("\n\n".join(page_rec_md_parts))

    if total_recovered_count > 0 and recovered_md_blocks:
        merged_rec_text = "\n\n".join(recovered_md_blocks)
        clean_md = f"{clean_md}\n\n## CONTENIDO Y LÍNEAS ADICIONALES RECUPERADAS:\n\n{merged_rec_text}".strip()
        print(f"[Box Recovery] ✅ ¡Rescate completado! Se recuperaron {total_recovered_count} líneas omitidas.", flush=True)
    else:
        print("[Box Recovery] ℹ️ Todos los elementos ya estaban presentes en el documento.", flush=True)

    return clean_md, sorted_pages, tokens
