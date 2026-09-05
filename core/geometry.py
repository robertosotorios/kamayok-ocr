"""
Utilidades geométricas para normalización de coordenadas, cálculo de bounding boxes y detección de extensiones.
"""
from typing import Dict, List, Optional, Tuple, Any

def detect_file_extension(file_bytes: bytes) -> str:
    """Detecta la extensión del archivo usando los números mágicos del encabezado."""
    if file_bytes.startswith(b"%PDF"):
        return ".pdf"
    if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if file_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    # WebP: Comienza con 'RIFF', seguido de 4 bytes de tamaño y luego 'WEBP'
    if file_bytes.startswith(b"RIFF") and len(file_bytes) >= 12 and file_bytes[8:12] == b"WEBP":
        return ".webp"
    if file_bytes.startswith(b"II*\x00") or file_bytes.startswith(b"MM\x00*"):
        return ".tiff"
    if file_bytes.startswith(b"BM"):
        return ".bmp"
    return ".pdf"  # Fallback por defecto

def compute_bounding_box(bboxes: List[Optional[Dict[str, float]]]) -> Optional[Dict[str, float]]:
    """Calcula el rectángulo envolvente (x1, y1, x2, y2) a partir de una lista de cajas."""
    valid_boxes = [b for b in bboxes if b]
    if not valid_boxes:
        return None
    return {
        "x1": round(min(b["x1"] for b in valid_boxes), 2),
        "y1": round(min(b["y1"] for b in valid_boxes), 2),
        "x2": round(max(b["x2"] for b in valid_boxes), 2),
        "y2": round(max(b["y2"] for b in valid_boxes), 2)
    }

def format_bbox(obj: Any, page_heights: Dict[int, Optional[float]], default_page: int = 1) -> Tuple[Optional[int], Optional[Dict[str, float]]]:
    """Convierte el bbox de Docling (de un prov, cell o bbox directo) a TOPLEFT (x1, y1, x2, y2)."""
    if not obj:
        return None, None

    b = None
    page_num = default_page

    # Si el objeto tiene atributo bbox (como Prov o TableCell)
    if hasattr(obj, "bbox") and obj.bbox is not None:
        b = obj.bbox
        page_num = getattr(obj, "page_no", default_page) or default_page
    # Si el objeto es directamente un BoundingBox con coordenadas l, t, r, b
    elif hasattr(obj, "l") and hasattr(obj, "t"):
        b = obj
        page_num = getattr(obj, "page_no", default_page) or default_page
    elif hasattr(obj, "prov") and obj.prov and len(obj.prov) > 0 and hasattr(obj.prov[0], "bbox"):
        b = obj.prov[0].bbox
        page_num = getattr(obj.prov[0], "page_no", default_page) or default_page

    if not b:
        return None, None

    page_h = page_heights.get(page_num)

    # 1. Utilizar método nativo de Docling si está disponible
    if page_h and hasattr(b, "to_top_left_origin") and callable(b.to_top_left_origin):
        try:
            b_tl = b.to_top_left_origin(page_h)
            return page_num, {
                "x1": round(min(b_tl.l, b_tl.r), 2),
                "y1": round(min(b_tl.t, b_tl.b), 2),
                "x2": round(max(b_tl.l, b_tl.r), 2),
                "y2": round(max(b_tl.t, b_tl.b), 2)
            }
        except Exception:
            pass

    # 2. Invertir coordenada vertical si el origen del documento es BOTTOMLEFT
    origin_str = str(getattr(b, "coord_origin", "")).upper()
    is_bottom_left = (
        "BOTTOMLEFT" in origin_str
        or getattr(getattr(b, "coord_origin", None), "name", "") == "BOTTOMLEFT"
        or (not origin_str and hasattr(b, "l") and hasattr(b, "b") and hasattr(b, "r") and hasattr(b, "t"))
    )

    if is_bottom_left and page_h:
        raw_x1, raw_y1 = b.l, page_h - b.t
        raw_x2, raw_y2 = b.r, page_h - b.b
    else:
        raw_x1, raw_y1 = b.l, b.t
        raw_x2, raw_y2 = b.r, b.b

    bbox = {
        "x1": round(min(raw_x1, raw_x2), 2),
        "y1": round(min(raw_y1, raw_y2), 2),
        "x2": round(max(raw_x1, raw_x2), 2),
        "y2": round(max(raw_y1, raw_y2), 2)
    }
    return page_num, bbox

def tokenize_text_to_spatial_tokens(
    text: str,
    bbox: Optional[Dict[str, float]],
    page: int,
    id_counter: Dict[str, int]
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Fase 1: Tokenización espacial inmutable.
    Divide una cadena en tokens atómicos (palabras, números, símbolos), asignando a cada
    uno un token_id único global inmutable (t_101, t_102...) y calculando su bounding box proporcional,
    con soporte consciente de líneas para evitar recuadros verticales en párrafos multilínea.
    """
    import re
    if not text or not text.strip():
        return [], []

    token_ids: List[str] = []
    tokens_list: List[Dict[str, Any]] = []

    has_bbox = bool(bbox and "x1" in bbox and "x2" in bbox and "y1" in bbox and "y2" in bbox)
    if not has_bbox:
        matches = list(re.finditer(r'\S+', text))
        for m in matches:
            token_str = m.group()
            if len(token_str) > 150 or "data:image" in token_str or token_str.startswith("data:"):
                continue
            id_counter["token"] = id_counter.get("token", 0) + 1
            tok_num = id_counter["token"]
            tok_id = f"t_{tok_num}"
            token_ids.append(tok_id)
            tokens_list.append({
                "id": tok_id,
                "text": token_str,
                "bbox": None,
                "page": page
            })
        return token_ids, tokens_list

    x1 = bbox["x1"]
    x2 = bbox["x2"]
    y1 = bbox["y1"]
    y2 = bbox["y2"]
    width = max(abs(x2 - x1), 1.0)
    height = max(abs(y2 - y1), 1.0)

    # 1. Determinar líneas de texto
    if "\n" in text:
        raw_lines = text.split("\n")
        lines = [l for l in raw_lines if l.strip()]
        if not lines:
            lines = [text]
    else:
        # Detectar si un bloque sin \n es multilínea según altura y caracteres
        raw_words = text.split()
        if len(raw_words) > 1:
            h_per_line = 48.0 if height > 100 else 14.0
            num_lines = max(1, min(len(raw_words), round(height / h_per_line)))
            min_char_w = 0.35 * (height / num_lines)
            max_chars_single_line = max(15, int(width / max(min_char_w, 1.0)))

            if num_lines > 1 and len(text) > max_chars_single_line * 0.8:
                # 1.1 Intentar partición por delimitadores semánticos naturales (códigos postales, prefijos viales, fiscales)
                LINE_BREAK_REGEX = re.compile(
                    r'^(?:D\.?N\.?I:?|C\.?I\.?F:?|N\.?I\.?F:?|NIE:?|VAT:?|\d{5}|'
                    r'Rúa|Rua|Calle|C/|Avda\.?|Avenida|Paseo|Plaza|Camino|Barrio|Carretera|Ctra\.?|'
                    r'Padre)$',
                    re.IGNORECASE
                )
                CONNECTORS = {"padre", "don", "de", "del", "la", "las", "el", "los", "san", "santa"}
                semantic_lines = []
                sem_curr = []
                for w in raw_words:
                    is_break = bool(LINE_BREAK_REGEX.match(w))
                    if is_break and sem_curr:
                        if sem_curr[-1].lower() in CONNECTORS:
                            sem_curr.append(w)
                            continue
                        semantic_lines.append(" ".join(sem_curr))
                        sem_curr = [w]
                    else:
                        sem_curr.append(w)
                if sem_curr:
                    semantic_lines.append(" ".join(sem_curr))

                # Si la partición semántica produjo al menos 2 líneas coherentes con la altura, adoptarla
                if len(semantic_lines) >= 2 and abs(len(semantic_lines) - num_lines) <= 1:
                    lines = semantic_lines
                else:
                    target_chars = len(text) / num_lines
                    lines = []
                    curr = []
                    curr_len = 0
                    for w in raw_words:
                        if curr and (curr_len + len(w) + 1 > target_chars * 1.15) and (len(lines) < num_lines - 1):
                            lines.append(" ".join(curr))
                            curr = [w]
                            curr_len = len(w)
                        else:
                            curr.append(w)
                            curr_len += len(w) + 1
                    if curr:
                        lines.append(" ".join(curr))
            else:
                lines = [text]
        else:
            lines = [text]

    num_lines = len(lines)
    line_h = height / num_lines
    max_line_chars = max(len(l) for l in lines) if lines else 1

    for l_idx, line_str in enumerate(lines):
        ly1 = round(y1 + l_idx * line_h, 2)
        ly2 = round(y1 + (l_idx + 1) * line_h, 2)
        line_len = max(len(line_str), 1)

        # Si una línea es significativamente más corta que la línea más ancha del bloque
        # (ej. "CIF: B-27750462" vs "INSTALACIONES Y OBRAS DE GALICIA, S.L."),
        # no estirar sus palabras por todo el ancho del bloque; usar su proporción real respecto a la línea máxima.
        eff_line_w = width if num_lines == 1 else min(width, width * (line_len / max(max_line_chars, 1)))

        for m in re.finditer(r'\S+', line_str):
            token_str = m.group()
            if len(token_str) > 150 or "data:image" in token_str or token_str.startswith("data:"):
                continue

            id_counter["token"] = id_counter.get("token", 0) + 1
            tok_num = id_counter["token"]
            tok_id = f"t_{tok_num}"
            token_ids.append(tok_id)

            tok_x1 = round(x1 + (m.start() / line_len) * eff_line_w, 2)
            tok_x2 = round(x1 + (m.end() / line_len) * eff_line_w, 2)

            tok_bbox = {
                "x1": min(tok_x1, tok_x2),
                "y1": ly1,
                "x2": max(tok_x1, tok_x2),
                "y2": ly2
            }

            tokens_list.append({
                "id": tok_id,
                "text": token_str,
                "bbox": tok_bbox,
                "page": page
            })

    return token_ids, tokens_list

