"""
Extracción estructurada de tablas jerárquicas (filas y celdas) y maquetación de páginas.
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional
from docling_core.types.doc.labels import DocItemLabel

from core.geometry import format_bbox, compute_bounding_box
from core.sanitizer import fix_split_accents

def extract_table_data(
    item: Any,
    block_bbox: Optional[Dict[str, float]],
    page_num: int,
    page_heights: Dict[int, Optional[float]],
    text_content: str
) -> Dict[str, Any]:
    """Extrae filas jerárquicas y celdas individuales con sus respectivos bounding boxes."""
    rows_dict = defaultdict(list)
    lines: List[Dict[str, Any]] = []

    for cell in item.data.table_cells:
        _, cell_bbox = format_bbox(cell, page_heights, default_page=page_num or 1)
        cell_text = fix_split_accents((cell.text or "").strip()) if getattr(cell, "text", None) else ""
        
        cell_dict = {
            "col_start": getattr(cell, "start_col_offset_idx", 0),
            "col_end": getattr(cell, "end_col_offset_idx", 0),
            "row_start": getattr(cell, "start_row_offset_idx", 0),
            "row_end": getattr(cell, "end_row_offset_idx", 0),
            "text": cell_text,
        }
        if cell_bbox:
            cell_dict["bbox"] = cell_bbox

        rows_dict[cell.start_row_offset_idx].append(cell_dict)

    structured_rows = []
    for r_idx in sorted(rows_dict.keys()):
        cells_in_row = sorted(rows_dict[r_idx], key=lambda c: c["col_start"])
        row_bbox = compute_bounding_box([c.get("bbox") for c in cells_in_row if c.get("bbox")])
        row_text = fix_split_accents(" | ".join(c["text"] for c in cells_in_row if c["text"]))

        row_dict = {
            "row_index": r_idx,
            "text": row_text,
            "cells": cells_in_row
        }
        if row_bbox:
            row_dict["bbox"] = row_bbox

        structured_rows.append(row_dict)

        # Cada fila de la tabla tiene su propio bbox de fila preciso
        if row_text and row_bbox:
            lines.append({
                "text": row_text,
                "bbox": row_bbox
            })

        # Cada celda individual tiene su propio bbox de celda
        for c in cells_in_row:
            if c.get("text") and c.get("bbox"):
                lines.append({
                    "text": c["text"],
                    "bbox": c["bbox"]
                })

    table_data = {
        "text": text_content,
        "bbox": block_bbox,
        "num_rows": getattr(item.data, "num_rows", len(structured_rows)),
        "num_cols": getattr(item.data, "num_cols", None),
        "rows": structured_rows
    }

    return {
        "table_data": table_data,
        "lines": lines
    }

def extract_layout_pages(
    doc: Any,
    tmp_path: Optional[str]
) -> List[Dict[str, Any]]:
    """Extrae la estructura de páginas y elementos de layout de Docling."""
    pages_dict: Dict[int, Dict[str, Any]] = {}
    page_heights: Dict[int, Optional[float]] = {}
    
    # 1. Mapear dimensiones de cada página
    if hasattr(doc, "pages") and doc.pages:
        for p_num, page in doc.pages.items():
            width = getattr(page.size, "width", None) or getattr(page.size, "w", None)
            height = getattr(page.size, "height", None) or getattr(page.size, "h", None)
            h_val = round(height, 2) if height is not None else None
            w_val = round(width, 2) if width is not None else None

            is_image_doc = bool(tmp_path and not tmp_path.lower().endswith(".pdf"))
            unit = "pixels" if is_image_doc else "points"

            page_heights[p_num] = h_val
            pages_dict[p_num] = {
                "page": p_num,
                "width": w_val,
                "height": h_val,
                "unit": unit,
                "elements": []
            }

    # 2. Extraer elementos de la maquetación
    for item, _ in doc.iterate_items():
        label_str = item.label.value if hasattr(item.label, "value") else str(item.label)
        
        # Omitir elementos gráficos puros (imágenes/fotos) para evitar ruido y cajas gigantes
        if label_str.lower() in ("picture", "figure") or getattr(item, "label", None) == DocItemLabel.PICTURE:
            continue

        text_content = getattr(item, "text", "")
        if not text_content and hasattr(item, "export_to_markdown"):
            try:
                text_content = item.export_to_markdown(doc=doc)
            except TypeError:
                try:
                    text_content = item.export_to_markdown()
                except Exception:
                    text_content = ""
            except Exception:
                text_content = ""

        text_content = fix_split_accents(text_content)

        # Si el contenido es un placeholder de imagen de Docling, descartarlo
        if not text_content or text_content.strip().startswith("<!-- 🖼️"):
            continue

        prov = item.prov[0] if getattr(item, "prov", None) else None
        page_num, block_bbox = format_bbox(prov, page_heights)
        current_page = page_num or 1

        table_data = None
        is_table = (
            getattr(item, "label", None) == DocItemLabel.TABLE or 
            str(getattr(item, "label", "")).lower() == "table"
        )

        lines: List[Dict[str, Any]] = []

        if is_table and hasattr(item, "data") and hasattr(item.data, "table_cells"):
            extracted = extract_table_data(item, block_bbox, current_page, page_heights, text_content)
            table_data = extracted["table_data"]
            lines = extracted["lines"]
        elif hasattr(item, "prov") and len(item.prov) > 1:
            # Párrafo o bloque con múltiples cajas sub-prov
            for sub_prov in item.prov:
                _, sub_bbox = format_bbox(sub_prov, page_heights)
                lines.append({
                    "text": fix_split_accents(getattr(sub_prov, "text", None) or text_content),
                    "bbox": sub_bbox
                })
        elif text_content:
            # Bloque de texto estándar
            lines.append({
                "text": text_content.strip(),
                "bbox": block_bbox
            })

        element_obj = {
            "label": label_str,
            "text": "" if is_table else text_content,
            "bbox": block_bbox,
            "lines": lines,
            "table_data": table_data
        }

        # Asociar el elemento a su página
        target_page = page_num if (page_num and page_num in pages_dict) else 1
        if target_page not in pages_dict:
            pages_dict[target_page] = {
                "page": target_page,
                "width": None,
                "height": None,
                "unit": "pixels" if (tmp_path and not tmp_path.lower().endswith(".pdf")) else "points",
                "elements": []
            }
        pages_dict[target_page]["elements"].append(element_obj)

    return [pages_dict[k] for k in sorted(pages_dict.keys())]

