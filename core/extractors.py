"""
Extracción estructurada de tablas jerárquicas (filas y celdas), clustering horizontal de líneas y maquetación de páginas.
"""
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from docling_core.types.doc.labels import DocItemLabel

from core.geometry import format_bbox, compute_bounding_box, tokenize_text_to_spatial_tokens
from core.sanitizer import fix_split_accents

# Regex para detectar formatos numéricos de importe (ej. 1.580,00, 331,80, 45.00, 38,8429, -4,1322, 100€)
PRICE_REGEX = re.compile(r'-?\b\d{1,3}(?:\.\d{3})*,\d{2,4}\s*€?\b|-?\b\d+\.\d{2,4}\s*€?\b')

def is_price_or_amount(text: str) -> bool:
    """Detecta si un texto contiene un formato de importe o precio con decimales."""
    return bool(PRICE_REGEX.search(text))

def match_line_item_pattern(text: str) -> bool:
    """Detecta si una línea de texto contiene una estructura de concepto + importe(s)."""
    has_letters = bool(re.search(r'[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]{3,}', text))
    has_price = is_price_or_amount(text)
    return has_letters and has_price

def cluster_horizontal_line_items(
    elements: List[Dict[str, Any]],
    id_counter: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Agrupa fragmentos de texto en la misma franja horizontal (Y similar) que formen
    líneas de factura con concepto e importes/precios (ej: 'Concepto' + 'Cant' + 'Precio' + 'Total').
    Preserva los token_ids inmutables de cada fragmento.
    """
    if not elements or len(elements) <= 1:
        return elements

    # Separar elementos de tablas formales (que ya vienen estructurados) de elementos de texto libre
    tables = [el for el in elements if el.get("label") == "table" or el.get("table_data")]
    non_tables = [el for el in elements if el.get("label") != "table" and not el.get("table_data")]

    if not non_tables:
        return elements

    # Filtrar elementos que tienen bbox válido
    with_bbox = [el for el in non_tables if el.get("bbox")]
    without_bbox = [el for el in non_tables if not el.get("bbox")]

    if not with_bbox:
        return elements

    used = set()
    clusters: List[List[Dict[str, Any]]] = []

    # Ordenar por coordenada vertical Y1 y luego horizontal X1
    sorted_els = sorted(with_bbox, key=lambda el: (el["bbox"]["y1"], el["bbox"]["x1"]))

    for i, el in enumerate(sorted_els):
        if i in used:
            continue
        
        current_cluster = [el]
        used.add(i)
        
        y1_curr = el["bbox"]["y1"]
        y2_curr = el["bbox"]["y2"]
        h_curr = y2_curr - y1_curr
        y_mid_curr = (y1_curr + y2_curr) / 2.0

        for j in range(i + 1, len(sorted_els)):
            if j in used:
                continue
            
            other = sorted_els[j]
            y1_other = other["bbox"]["y1"]
            y2_other = other["bbox"]["y2"]
            y_mid_other = (y1_other + y2_other) / 2.0

            # Si la distancia vertical es mayor a la altura del elemento, ya no puede estar en la misma línea
            if abs(y_mid_curr - y_mid_other) > max(8.0, h_curr * 0.5):
                continue

            # Comprobar solapamiento vertical
            overlap = min(y2_curr, y2_other) - max(y1_curr, y1_other)
            min_h = min(h_curr, y2_other - y1_other)
            
            if (min_h > 0 and overlap / min_h >= 0.4) or abs(y_mid_curr - y_mid_other) <= 6.0:
                current_cluster.append(other)
                used.add(j)

        clusters.append(current_cluster)

    processed_elements: List[Dict[str, Any]] = []

    for cluster in clusters:
        if len(cluster) == 1:
            processed_elements.append(cluster[0])
            continue

        # Ordenar los elementos del cluster de izquierda a derecha por X1
        sorted_cluster = sorted(cluster, key=lambda el: el["bbox"]["x1"])
        combined_text = " | ".join(el["text"].strip() for el in sorted_cluster if el.get("text", "").strip())
        
        # Verificar si la fila combinada contiene un patrón de concepto + importe
        if match_line_item_pattern(combined_text):
            row_bbox = compute_bounding_box([el["bbox"] for el in sorted_cluster])
            sub_lines: List[Dict[str, Any]] = []
            combined_token_ids: List[str] = []
            
            for el in sorted_cluster:
                if el.get("token_ids"):
                    combined_token_ids.extend(el["token_ids"])
                if el.get("text") and el.get("bbox"):
                    if el.get("lines"):
                        sub_lines.extend(el["lines"])
                    else:
                        sub_lines.append({
                            "line_id": el.get("block_id") or f"l_{id_counter.get('line', 0)}",
                            "text": el["text"].strip(),
                            "bbox": el["bbox"],
                            "token_ids": el.get("token_ids", [])
                        })

            id_counter["block"] = id_counter.get("block", 0) + 1
            line_item_obj = {
                "block_id": f"b_{id_counter['block']}",
                "label": "line_item",
                "text": combined_text,
                "bbox": row_bbox,
                "lines": sub_lines,
                "token_ids": combined_token_ids,
                "table_data": None
            }
            processed_elements.append(line_item_obj)
        else:
            processed_elements.extend(sorted_cluster)

    # Recombinar con tablas y elementos sin bbox, preservando orden
    all_final_elements = tables + processed_elements + without_bbox
    return all_final_elements

def extract_table_data(
    item: Any,
    block_bbox: Optional[Dict[str, float]],
    page_num: int,
    page_heights: Dict[int, Optional[float]],
    id_counter: Dict[str, int],
    all_tokens_collector: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Fase 2: Reconstrucción estructural de tablas jerárquicas con Celdas y Filas.
    Asigna IDs inmutables a cada Celda (cell_id), Fila (row_id) y asocia sus listas de token_ids.
    """
    rows_dict = defaultdict(list)
    lines: List[Dict[str, Any]] = []
    table_token_ids: List[str] = []

    for cell in item.data.table_cells:
        _, cell_bbox = format_bbox(cell, page_heights, default_page=page_num or 1)
        cell_text = fix_split_accents((cell.text or "").strip()) if getattr(cell, "text", None) else ""
        
        # Tokenizar la celda y recolectar sus tokens inmutables
        c_tok_ids, c_tokens = tokenize_text_to_spatial_tokens(cell_text, cell_bbox, page_num, id_counter)
        all_tokens_collector.extend(c_tokens)
        table_token_ids.extend(c_tok_ids)

        id_counter["cell"] = id_counter.get("cell", 0) + 1
        cell_id = f"c_{id_counter['cell']}"

        cell_dict = {
            "cell_id": cell_id,
            "col_start": getattr(cell, "start_col_offset_idx", 0),
            "col_end": getattr(cell, "end_col_offset_idx", 0),
            "row_start": getattr(cell, "start_row_offset_idx", 0),
            "row_end": getattr(cell, "end_row_offset_idx", 0),
            "text": cell_text,
            "token_ids": c_tok_ids,
        }
        if cell_bbox:
            cell_dict["bbox"] = cell_bbox

        rows_dict[cell.start_row_offset_idx].append(cell_dict)

    structured_rows = []
    for r_idx in sorted(rows_dict.keys()):
        cells_in_row = sorted(rows_dict[r_idx], key=lambda c: c["col_start"])
        row_bbox = compute_bounding_box([c.get("bbox") for c in cells_in_row if c.get("bbox")])
        row_text = fix_split_accents(" | ".join(c["text"] for c in cells_in_row if c["text"]))
        
        row_token_ids: List[str] = []
        for c in cells_in_row:
            row_token_ids.extend(c.get("token_ids", []))

        id_counter["row"] = id_counter.get("row", 0) + 1
        row_id = f"r_{id_counter['row']}"

        row_dict = {
            "row_id": row_id,
            "row_index": r_idx,
            "text": row_text,
            "cells": cells_in_row,
            "token_ids": row_token_ids
        }
        if row_bbox:
            row_dict["bbox"] = row_bbox

        structured_rows.append(row_dict)

        # Fila de tabla en lines
        if row_text and row_bbox:
            lines.append({
                "line_id": row_id,
                "text": row_text,
                "bbox": row_bbox,
                "token_ids": row_token_ids
            })

        # Celdas individuales en lines
        for c in cells_in_row:
            if c.get("text") and c.get("bbox"):
                lines.append({
                    "line_id": c["cell_id"],
                    "text": c["text"],
                    "bbox": c["bbox"],
                    "token_ids": c.get("token_ids", [])
                })

    table_data = {
        "bbox": block_bbox,
        "num_rows": getattr(item.data, "num_rows", len(structured_rows)),
        "num_cols": getattr(item.data, "num_cols", None),
        "rows": structured_rows,
        "token_ids": table_token_ids
    }

    return {
        "table_data": table_data,
        "lines": lines,
        "token_ids": table_token_ids
    }

def extract_layout_pages(
    doc: Any,
    tmp_path: Optional[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extrae la estructura de páginas, elementos de layout, y el catálogo plano de tokens
    espaciales inmutables (Fase 1 y Fase 2).
    Retorna (sorted_pages, all_tokens).
    """
    pages_dict: Dict[int, Dict[str, Any]] = {}
    page_heights: Dict[int, Optional[float]] = {}
    id_counter: Dict[str, int] = {"token": 0, "cell": 0, "row": 0, "line": 0, "block": 0}
    all_tokens: List[Dict[str, Any]] = []
    
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

    # 2. Extraer elementos de la maquetación (con soporte de traverse_pictures para no perder texto en cajas gráficas)
    try:
        items_iterator = doc.iterate_items(traverse_pictures=True)
    except TypeError:
        items_iterator = doc.iterate_items()

    for item, _ in items_iterator:
        label_str = item.label.value if hasattr(item.label, "value") else str(item.label)
        is_picture = (
            label_str.lower() in ("picture", "figure") or 
            getattr(item, "label", None) == DocItemLabel.PICTURE
        )

        text_content = getattr(item, "text", "")
        if not text_content and hasattr(item, "caption_text"):
            try:
                text_content = item.caption_text(doc=doc)
            except Exception:
                text_content = ""

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

        text_content = fix_split_accents(text_content or "")

        # Si el contenido contiene imágenes en base64 o enlaces de imagen Markdown, descartarlo
        if "data:image" in text_content or text_content.strip().startswith("!["):
            text_content = ""

        # Si es un contenedor de imagen y carece de texto real legible o es un placeholder gráfico, descartarlo
        is_placeholder = (
            not text_content or 
            text_content.strip().startswith("<!--") or 
            text_content.strip().startswith("![") or
            "data:image" in text_content
        )

        is_table = (
            getattr(item, "label", None) == DocItemLabel.TABLE or 
            str(getattr(item, "label", "")).lower() == "table"
        )
        has_table_cells = is_table and hasattr(item, "data") and hasattr(item.data, "table_cells")

        if is_placeholder and not has_table_cells:
            continue

        # Si un elemento gráfico contenía texto real, reclasificar su etiqueta como texto para maquetación
        if is_picture and text_content and not is_placeholder:
            label_str = "text"

        prov = item.prov[0] if getattr(item, "prov", None) else None
        page_num, block_bbox = format_bbox(prov, page_heights)
        current_page = page_num or 1
        table_data = None

        lines: List[Dict[str, Any]] = []
        element_token_ids: List[str] = []

        if is_table and hasattr(item, "data") and hasattr(item.data, "table_cells"):
            extracted = extract_table_data(item, block_bbox, current_page, page_heights, id_counter, all_tokens)
            table_data = extracted["table_data"]
            lines = extracted["lines"]
            element_token_ids = extracted.get("token_ids", [])
        elif hasattr(item, "prov") and len(item.prov) > 1:
            # Párrafo o bloque con múltiples cajas sub-prov
            for sub_prov in item.prov:
                _, sub_bbox = format_bbox(sub_prov, page_heights)
                sub_text = fix_split_accents(getattr(sub_prov, "text", None) or text_content)
                s_tok_ids, s_tokens = tokenize_text_to_spatial_tokens(sub_text, sub_bbox, current_page, id_counter)
                all_tokens.extend(s_tokens)
                element_token_ids.extend(s_tok_ids)

                id_counter["line"] = id_counter.get("line", 0) + 1
                lines.append({
                    "line_id": f"l_{id_counter['line']}",
                    "text": sub_text,
                    "bbox": sub_bbox,
                    "token_ids": s_tok_ids
                })
        elif text_content:
            # Bloque de texto estándar
            clean_txt = text_content.strip()
            b_tok_ids, b_tokens = tokenize_text_to_spatial_tokens(clean_txt, block_bbox, current_page, id_counter)
            all_tokens.extend(b_tokens)
            element_token_ids.extend(b_tok_ids)

            id_counter["line"] = id_counter.get("line", 0) + 1
            lines.append({
                "line_id": f"l_{id_counter['line']}",
                "text": clean_txt,
                "bbox": block_bbox,
                "token_ids": b_tok_ids
            })

        id_counter["block"] = id_counter.get("block", 0) + 1
        block_id = f"b_{id_counter['block']}"
        table_text = "\n".join(l["text"] for l in lines if l.get("text")) if lines else text_content
        element_obj = {
            "block_id": block_id,
            "label": label_str,
            "text": (table_text or text_content or "") if is_table else text_content,
            "bbox": block_bbox,
            "lines": lines,
            "token_ids": element_token_ids,
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

    # 3. Aplicar clustering horizontal a las líneas sueltas de cada página para reconstruir line items
    for p_num in pages_dict:
        pages_dict[p_num]["elements"] = cluster_horizontal_line_items(pages_dict[p_num]["elements"], id_counter)

    sorted_pages = [pages_dict[k] for k in sorted(pages_dict.keys())]
    return sorted_pages, all_tokens


