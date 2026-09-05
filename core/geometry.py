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

    # Invertir coordenada vertical si el origen del documento es BOTTOMLEFT
    if str(getattr(b, "coord_origin", "")).upper() == "BOTTOMLEFT" and page_h:
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
    uno un token_id único global inmutable (t_101, t_102...) y calculando su bounding box proporcional.
    """
    import re
    if not text or not text.strip():
        return [], []

    token_ids: List[str] = []
    tokens_list: List[Dict[str, Any]] = []

    matches = list(re.finditer(r'\S+', text))
    if not matches:
        return [], []

    total_len = max(len(text), 1)

    for m in matches:
        token_str = m.group()
        # Ignorar cadenas base64 o tokens espurios excesivamente largos
        if len(token_str) > 150 or "data:image" in token_str or token_str.startswith("data:"):
            continue
        id_counter["token"] = id_counter.get("token", 0) + 1
        tok_id = f"t_{id_counter['token']}"
        token_ids.append(tok_id)

        tok_bbox = None
        if bbox and "x1" in bbox and "x2" in bbox:
            x1 = bbox["x1"]
            x2 = bbox["x2"]
            y1 = bbox["y1"]
            y2 = bbox["y2"]
            width = x2 - x1
            tok_x1 = round(x1 + (m.start() / total_len) * width, 2)
            tok_x2 = round(x1 + (m.end() / total_len) * width, 2)
            tok_bbox = {
                "x1": min(tok_x1, tok_x2),
                "y1": y1,
                "x2": max(tok_x1, tok_x2),
                "y2": y2
            }

        tokens_list.append({
            "id": tok_id,
            "text": token_str,
            "bbox": tok_bbox,
            "page": page
        })

    return token_ids, tokens_list

