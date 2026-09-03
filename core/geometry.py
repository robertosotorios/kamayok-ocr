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

