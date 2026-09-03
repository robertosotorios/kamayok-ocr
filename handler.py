import os
import base64
import tempfile
import torch
import cv2
import numpy as np
from collections import defaultdict
import runpod

from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
    EasyOcrOptions,
)
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.labels import DocItemLabel

# 0. Diagnóstico y verificación de GPU
cuda_available = torch.cuda.is_available()
device = AcceleratorDevice.CUDA if cuda_available else AcceleratorDevice.CPU
print(f"[Docling Worker] CUDA Available: {cuda_available}")
if cuda_available:
    print(f"[Docling Worker] GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"[Docling Worker] GPU Count: {torch.cuda.device_count()}")
    print(f"[Docling Worker] VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("[Docling Worker] ⚠️ ADVERTENCIA: CUDA no detectado. Ejecutando en CPU fallback.")

# 1. Configuración de Pipeline con aceleración completa por GPU
pipeline_options = PdfPipelineOptions()
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4,
    device=device
)
pipeline_options.do_table_structure = True
pipeline_options.do_ocr = True

# Forzar el motor de OCR a utilizar la GPU si está disponible
if cuda_available:
    try:
        pipeline_options.ocr_options = EasyOcrOptions(use_gpu=True, lang=["es", "en"])
    except Exception as e:
        print(f"[Docling Worker] Nota: Configurando OCR por defecto con GPU ({e})")
else:
    try:
        pipeline_options.ocr_options = EasyOcrOptions(use_gpu=False, lang=["es", "en"])
    except Exception as e:
        pass

# 2. Inicialización de conversores soportando PDF y formatos de imagen (incluyendo WebP)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options)
    }
)

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

def compute_bounding_box(bboxes):
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

def format_bbox(obj, page_heights, default_page=1):
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

def extract_qr_codes(doc, tmp_path):
    """Detecta y decodifica códigos QR (VeriFactu, TicketBAI, URLs de pago) en PDFs e imágenes."""
    qr_results = []
    detector = cv2.QRCodeDetector()

    def process_image(img_bgr, page_num=1):
        if img_bgr is None or img_bgr.size == 0:
            return
        try:
            success, decoded_info, points, _ = detector.detectAndDecodeMulti(img_bgr)
            if success and decoded_info:
                for text in decoded_info:
                    if text and text.strip():
                        val = text.strip()
                        if not any(q["value"] == val for q in qr_results):
                            is_verifactu = "agenciatributaria.gob.es" in val.lower() or "verifactu" in val.lower()
                            is_ticketbai = "tbai" in val.lower() or "ticketbai" in val.lower() or "gipuzkoa.eus" in val.lower() or "bizkaia.eus" in val.lower() or "araba.eus" in val.lower()
                            
                            qr_results.append({
                                "value": val,
                                "page": page_num,
                                "is_verifactu": is_verifactu,
                                "is_ticketbai": is_ticketbai,
                                "type": "verifactu" if is_verifactu else ("ticketbai" if is_ticketbai else "qr_code")
                            })
            else:
                text, pts, _ = detector.detectAndDecode(img_bgr)
                if text and text.strip():
                    val = text.strip()
                    if not any(q["value"] == val for q in qr_results):
                        is_verifactu = "agenciatributaria.gob.es" in val.lower() or "verifactu" in val.lower()
                        is_ticketbai = "tbai" in val.lower() or "ticketbai" in val.lower() or "gipuzkoa.eus" in val.lower() or "bizkaia.eus" in val.lower() or "araba.eus" in val.lower()
                        
                        qr_results.append({
                            "value": val,
                            "page": page_num,
                            "is_verifactu": is_verifactu,
                            "is_ticketbai": is_ticketbai,
                            "type": "verifactu" if is_verifactu else ("ticketbai" if is_ticketbai else "qr_code")
                        })
        except Exception:
            pass

    # 1. Si es PDF, renderizar cada página con pypdfium2 para máxima precisión
    if tmp_path and tmp_path.lower().endswith(".pdf"):
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(tmp_path)
            for page_idx in range(len(pdf)):
                page = pdf[page_idx]
                bitmap = page.render(scale=2)
                pil_image = bitmap.to_pil()
                img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                process_image(img_np, page_num=page_idx + 1)
        except Exception as e:
            print(f"[Docling Worker] Escaneo QR en PDF: {e}")

    # 2. Si es una imagen (PNG, JPG, WebP)
    elif tmp_path and os.path.exists(tmp_path):
        try:
            img = cv2.imread(tmp_path)
            process_image(img, page_num=1)
        except Exception as e:
            print(f"[Docling Worker] Escaneo QR en imagen: {e}")

    return qr_results

def handler(event):
    job_input = event.get("input", {})
    
    # Soporte para claves directas o genéricas
    file_url = job_input.get("file_url") or job_input.get("pdf_url") or job_input.get("image_url")
    file_base64 = job_input.get("file_base64") or job_input.get("pdf_base64") or job_input.get("image_base64")

    if not file_url and not file_base64:
        return {
            "error": "Debes proporcionar un archivo mediante URL ('file_url', 'pdf_url') o Base64 ('file_base64', 'pdf_base64')",
            "status": "failed"
        }

    tmp_path = None
    try:
        if file_url:
            source = file_url
        else:
            file_bytes = base64.b64decode(file_base64)
            ext = detect_file_extension(file_bytes)
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            source = tmp_path

        # Procesar con Docling
        result = converter.convert(source)
        doc = result.document

        # 1. Mapear dimensiones de cada página/imagen
        pages_dict = {}
        page_heights = {}
        
        if hasattr(doc, "pages") and doc.pages:
            for p_num, page in doc.pages.items():
                width = getattr(page.size, "width", None) or getattr(page.size, "w", None)
                height = getattr(page.size, "height", None) or getattr(page.size, "h", None)
                h_val = round(height, 2) if height is not None else None
                w_val = round(width, 2) if width is not None else None

                # Determinar unidad: 'pixels' si es imagen o no tiene dimensiones en puntos estándar
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

            # Si el contenido es un placeholder de imagen de Docling, descartarlo
            if not text_content or text_content.strip().startswith("<!-- 🖼️"):
                continue

            prov = item.prov[0] if getattr(item, "prov", None) else None
            page_num, block_bbox = format_bbox(prov, page_heights)

            # 2.1 Extracción jerárquica de tablas
            table_data = None
            is_table = (
                getattr(item, "label", None) == DocItemLabel.TABLE or 
                str(getattr(item, "label", "")).lower() == "table"
            )

            lines = []

            if is_table and hasattr(item, "data") and hasattr(item.data, "table_cells"):
                rows_dict = defaultdict(list)
                for cell in item.data.table_cells:
                    _, cell_bbox = format_bbox(cell, page_heights, default_page=page_num or 1)

                    cell_dict = {
                        "col_start": getattr(cell, "start_col_offset_idx", 0),
                        "col_end": getattr(cell, "end_col_offset_idx", 0),
                        "row_start": getattr(cell, "start_row_offset_idx", 0),
                        "row_end": getattr(cell, "end_row_offset_idx", 0),
                        "text": (cell.text or "").strip() if getattr(cell, "text", None) else "",
                    }
                    if cell_bbox:
                        cell_dict["bbox"] = cell_bbox

                    rows_dict[cell.start_row_offset_idx].append(cell_dict)

                structured_rows = []
                for r_idx in sorted(rows_dict.keys()):
                    cells_in_row = sorted(rows_dict[r_idx], key=lambda c: c["col_start"])
                    row_bbox = compute_bounding_box([c.get("bbox") for c in cells_in_row if c.get("bbox")])
                    row_text = " | ".join(c["text"] for c in cells_in_row if c["text"])

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

            elif hasattr(item, "prov") and len(item.prov) > 1:
                # Párrafo o bloque con múltiples cajas sub-prov
                for sub_prov in item.prov:
                    _, sub_bbox = format_bbox(sub_prov, page_heights)
                    lines.append({
                        "text": getattr(sub_prov, "text", None) or text_content,
                        "bbox": sub_bbox
                    })
            elif text_content:
                # Bloque de texto estándar
                lines.append({
                    "text": text_content.strip(),
                    "bbox": block_bbox
                })

            element_obj = {
                "label": item.label.value if hasattr(item.label, "value") else str(item.label),
                "text": text_content,
                "bbox": block_bbox,
                "lines": lines,
                "table_data": table_data
            }

            # Asociar el elemento a su página (o crear la página 1 si no existía el registro previo)
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

        sorted_pages = [pages_dict[k] for k in sorted(pages_dict.keys())]

        # Limpiar etiquetas de imágenes residuales en el Markdown
        raw_md = doc.export_to_markdown()
        clean_md = raw_md.replace("<!-- image -->", "").replace("<!-- 🖼️❌ Image not available. Please use `PdfPipelineOptions(generate_picture_images=True)` -->", "")
        import re
        clean_md = re.sub(r'\n{3,}', '\n\n', clean_md).strip()

        return {
            "status": "success",
            "markdown": clean_md,
            "qr_codes": qr_codes,
            "pages": sorted_pages
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# =============================================================================
# Soporte para RunPod Load Balancer (HTTP Server + /ping) y Serverless clásico
# =============================================================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Docling RunPod Service")

@app.get("/ping")
@app.get("/health")
def ping():
    return {
        "status": "healthy",
        "cuda_available": cuda_available,
        "gpu_model": torch.cuda.get_device_name(0) if cuda_available else "CPU"
    }

@app.post("/runsync")
@app.post("/")
async def runsync_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    result = handler(body)
    return {
        "status": "COMPLETED",
        "output": result
    }

@app.post("/run")
async def run_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    result = handler(body)
    return {
        "id": "job-direct",
        "status": "COMPLETED",
        "output": result
    }

if __name__ == "__main__":
    port_env = os.environ.get("PORT")
    port_health_env = os.environ.get("PORT_HEALTH")

    # Si RunPod se ejecuta como Load Balancer o tiene la variable PORT configurada
    if port_env or port_health_env or os.environ.get("RUNPOD_HTTP_SERVER") == "1":
        port = int(port_env or port_health_env or 8000)
        print(f"[Docling Worker] 🚀 Servidor HTTP Load Balancer activo en puerto {port} (Endpoint de salud: /ping)")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # Modo RunPod Serverless estándar (Job Queue)
        print("[Docling Worker] 🚀 RunPod Serverless Worker activo (Job Queue)")
        runpod.serverless.start({"handler": handler})
