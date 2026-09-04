"""
Módulo de recuperación de texto marginal y vertical (CIF, NIF, datos registrales).
Garantiza que ningún texto rotado a 90°/270° o situado en los bordes del documento sea omitido por Docling ni por el OCR.
"""
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from core.geometry import compute_bounding_box
from core.sanitizer import fix_split_accents, fix_fiscal_identifiers

# Expresión regular para detectar identificadores fiscales, mercantiles o corporativos
FISCAL_REGEX = re.compile(
    r'\b(?:CIF|NIF|VAT|DNI|NIE|RUT|CNPJ|ES[A-Z0-9]{8,9}|[A-Z]\d{7,8}[A-Z0-9]?|'
    r'Registro\s+Mercantil|Tomo|Folio|Sección|Hoja|Inscripción|S\.L\.|S\.A\.|S\.L\.U\.|'
    r'Paseo|Avenida|Avda|Calle|C/|Rúa|Edificio|Polígono)\b',
    re.IGNORECASE
)

def is_text_already_in_markdown(text: str, markdown: str) -> bool:
    """Comprueba si una porción significativa del texto ya está presente en el markdown."""
    if not text or not markdown:
        return False
    # Normalizar eliminando espacios y signos de puntuación
    clean_t = re.sub(r'\W+', '', text.lower())
    clean_m = re.sub(r'\W+', '', markdown.lower())
    if len(clean_t) < 8:
        return clean_t in clean_m
    # Buscar subsecuencias de 20 caracteres
    return (clean_t[:20] in clean_m) or (clean_t[-20:] in clean_m)

def recover_native_pdf_margins(
    pdf_path: str,
    sorted_pages: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
    clean_md: str,
    id_counter: Dict[str, int]
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Recupera texto nativo del PDF (por ejemplo, textos verticales en márgenes o pie de página)
    que Docling haya omitido debido a la orientación o clasificación de maquetación.
    """
    try:
        cmd = ["pdftotext", "-bbox", pdf_path, "-"]
        xml_data = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[Docling Worker] ⚠️ Advertencia en pdftotext -bbox: {e}", flush=True)
        return clean_md, sorted_pages, tokens

    pages_xml = re.findall(r'<page width=\"([0-9.]+)\" height=\"([0-9.]+)\">(.*?)</page>', xml_data, re.DOTALL)
    if not pages_xml:
        return clean_md, sorted_pages, tokens

    page_map = {p.get("page", idx + 1): p for idx, p in enumerate(sorted_pages)}
    recovered_headers = []

    for p_idx, (w_str, h_str, p_content) in enumerate(pages_xml):
        page_num = p_idx + 1
        page_w = float(w_str)
        page_h = float(h_str)

        raw_words = re.findall(
            r'<word xMin=\"([0-9.]+)\" yMin=\"([0-9.]+)\" xMax=\"([0-9.]+)\" yMax=\"([0-9.]+)\">(.*?)</word>',
            p_content
        )
        if not raw_words:
            continue

        # Extraer palabras situadas en los márgenes izquierdo (< 35 pt) o derecho (> page_w - 35 pt)
        margin_words_left = []
        margin_words_right = []

        for x1, y1, x2, y2, text in raw_words:
            w_obj = {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "text": text.strip()
            }
            if w_obj["x2"] <= 35.0:
                margin_words_left.append(w_obj)
            elif w_obj["x1"] >= (page_w - 35.0):
                margin_words_right.append(w_obj)

        for side, m_words in [("left", margin_words_left), ("right", margin_words_right)]:
            if not m_words:
                continue

            full_line_text = fix_fiscal_identifiers(fix_split_accents(" ".join(w["text"] for w in m_words if w["text"])))
            if not full_line_text.strip():
                continue

            if not is_text_already_in_markdown(full_line_text, clean_md):
                print(f"[Docling Worker] 🔍 Recuperando texto marginal en página {page_num} ({side}): {full_line_text[:80]}...", flush=True)

                bbox = {
                    "x1": round(min(w["x1"] for w in m_words), 2),
                    "y1": round(min(w["y1"] for w in m_words), 2),
                    "x2": round(max(w["x2"] for w in m_words), 2),
                    "y2": round(max(w["y2"] for w in m_words), 2),
                }

                line_tok_ids = []
                line_sub_lines = []
                for w in m_words:
                    id_counter["token"] = id_counter.get("token", 0) + 1
                    tok_id = f"t_{id_counter['token']}"
                    line_tok_ids.append(tok_id)
                    tokens.append({
                        "id": tok_id,
                        "text": w["text"],
                        "bbox": {
                            "x1": round(w["x1"], 2),
                            "y1": round(w["y1"], 2),
                            "x2": round(w["x2"], 2),
                            "y2": round(w["y2"], 2)
                        },
                        "page": page_num
                    })

                id_counter["line"] = id_counter.get("line", 0) + 1
                line_id = f"l_{id_counter['line']}"
                line_sub_lines.append({
                    "line_id": line_id,
                    "text": full_line_text,
                    "bbox": bbox,
                    "token_ids": line_tok_ids
                })

                id_counter["block"] = id_counter.get("block", 0) + 1
                block_id = f"b_m_{id_counter['block']}"

                element_obj = {
                    "block_id": block_id,
                    "label": "margin_text",
                    "text": full_line_text,
                    "bbox": bbox,
                    "lines": line_sub_lines,
                    "token_ids": line_tok_ids,
                    "table_data": None
                }

                if page_num in page_map:
                    page_map[page_num]["elements"].insert(0, element_obj)
                else:
                    new_p = {
                        "page": page_num,
                        "width": page_w,
                        "height": page_h,
                        "unit": "points",
                        "elements": [element_obj]
                    }
                    sorted_pages.append(new_p)
                    page_map[page_num] = new_p

                recovered_headers.append(full_line_text)

    if recovered_headers:
        header_block = "\n\n".join(recovered_headers)
        clean_md = f"{header_block}\n\n{clean_md}".strip()
        print(f"[Docling Worker] ✅ Inyectadas {len(recovered_headers)} línea(s) marginal(es) en Markdown y tokens.", flush=True)

    return clean_md, sorted_pages, tokens


def recover_scanned_image_margins(
    img_path: str,
    sorted_pages: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
    clean_md: str,
    id_counter: Dict[str, int]
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Para documentos escaneados o imágenes: recorta los márgenes verticales izquierdo y derecho,
    los rota 90°/270° y ejecuta OCR para rescatar CIFs o identificadores impresos verticalmente.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return clean_md, sorted_pages, tokens

    try:
        pil_img = Image.open(img_path)
    except Exception:
        return clean_md, sorted_pages, tokens

    img_w, img_h = pil_img.size
    # Analizar el 10% del ancho en los bordes izquierdo y derecho
    margin_w = max(int(img_w * 0.10), 40)
    strips = [
        ("left", (0, 0, margin_w, img_h)),
        ("right", (img_w - margin_w, 0, img_w, img_h))
    ]

    # Cargar RapidOCR u OCR disponible
    ocr_engine = None
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr_engine = RapidOCR()
    except Exception:
        pass

    if not ocr_engine:
        return clean_md, sorted_pages, tokens

    recovered_lines = []
    page_num = 1
    page_map = {p.get("page", idx + 1): p for idx, p in enumerate(sorted_pages)}

    for side, crop_box in strips:
        try:
            strip_crop = pil_img.crop(crop_box)
            # Probar rotación de 270° (texto vertical ascendente típico) y 90° (descendente)
            for rot_angle in [270, 90]:
                rot_img = strip_crop.rotate(rot_angle, expand=True)
                rot_np = np.array(rot_img)
                ocr_res, _ = ocr_engine(rot_np)
                if not ocr_res:
                    continue

                full_text = " ".join([r[1] for r in ocr_res if r and len(r) > 1 and r[1]]).strip()
                full_text = fix_fiscal_identifiers(fix_split_accents(full_text))

                if full_text and FISCAL_REGEX.search(full_text):
                    if not is_text_already_in_markdown(full_text, clean_md):
                        print(f"[Docling Worker] 🔍 Recuperado texto vertical de margen escaneado ({side}, rot {rot_angle}°): {full_text[:80]}...", flush=True)
                        
                        id_counter["block"] = id_counter.get("block", 0) + 1
                        block_id = f"b_m_{id_counter['block']}"

                        bbox = {
                            "x1": float(crop_box[0]),
                            "y1": float(crop_box[1]),
                            "x2": float(crop_box[2]),
                            "y2": float(crop_box[3])
                        }

                        line_tok_ids = []
                        line_words = full_text.split()
                        for w in line_words:
                            id_counter["token"] = id_counter.get("token", 0) + 1
                            tok_id = f"t_{id_counter['token']}"
                            line_tok_ids.append(tok_id)
                            tokens.append({
                                "id": tok_id,
                                "text": w,
                                "bbox": bbox,
                                "page": page_num
                            })

                        id_counter["line"] = id_counter.get("line", 0) + 1
                        line_id = f"l_{id_counter['line']}"

                        element_obj = {
                            "block_id": block_id,
                            "label": "margin_text",
                            "text": full_text,
                            "bbox": bbox,
                            "lines": [{
                                "line_id": line_id,
                                "text": full_text,
                                "bbox": bbox,
                                "token_ids": line_tok_ids
                            }],
                            "token_ids": line_tok_ids,
                            "table_data": None
                        }

                        if page_num in page_map:
                            page_map[page_num]["elements"].insert(0, element_obj)
                        else:
                            new_p = {
                                "page": page_num,
                                "width": float(img_w),
                                "height": float(img_h),
                                "unit": "pixels",
                                "elements": [element_obj]
                            }
                            sorted_pages.append(new_p)
                            page_map[page_num] = new_p

                        recovered_lines.append(full_text)
                        break  # Si se detectó en 270°, no es necesario 90°
        except Exception as strip_err:
            print(f"[Docling Worker] ⚠️ Error en escaneo de margen {side}: {strip_err}", flush=True)

    if recovered_lines:
        header_block = "\n\n".join(recovered_lines)
        clean_md = f"{header_block}\n\n{clean_md}".strip()
        print(f"[Docling Worker] ✅ Inyectadas {len(recovered_lines)} línea(s) de margen escaneado en Markdown y tokens.", flush=True)

    return clean_md, sorted_pages, tokens


def recover_missing_margin_content(
    sorted_pages: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
    clean_md: str,
    source_path: str,
    id_counter: Dict[str, int]
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Punto de entrada unificado para recuperar márgenes omitidos tanto en PDFs nativos como en escaneados/imágenes.
    """
    if not source_path or not os.path.exists(source_path):
        return clean_md, sorted_pages, tokens

    if source_path.lower().endswith(".pdf"):
        # 1. Recuperar texto nativo de márgenes desde el flujo PDF
        clean_md, sorted_pages, tokens = recover_native_pdf_margins(
            source_path, sorted_pages, tokens, clean_md, id_counter
        )
    else:
        # 2. Si es imagen directa (o PDF rasterizado a imagen), ejecutar escaneo rotado de márgenes
        clean_md, sorted_pages, tokens = recover_scanned_image_margins(
            source_path, sorted_pages, tokens, clean_md, id_counter
        )

    return clean_md, sorted_pages, tokens

