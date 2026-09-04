"""
Servicio OCR Docling + EasyOCR optimizado para RunPod Serverless (Job Queue / Webhook).
"""
import os
import sys
import base64
import tempfile
import traceback
import requests
import torch
import runpod

from core.config import converter, cpu_converter, cuda_available
from core.geometry import detect_file_extension
from core.sanitizer import sanitize_docling_document, clean_markdown_output
from core.extractors import extract_layout_pages
from core.qr_scanner import extract_qr_codes

def handler(event):
    """Manejador principal de peticiones OCR de RunPod Serverless."""
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
            print(f"[Docling Worker] 📥 Descargando archivo desde URL: {file_url[:80]}...", flush=True)
            res = requests.get(file_url, stream=True, timeout=120)
            res.raise_for_status()
            file_bytes = res.content
        else:
            print(f"[Docling Worker] 📦 Decodificando archivo desde Base64 ({len(file_base64)} chars)...", flush=True)
            file_bytes = base64.b64decode(file_base64)

        ext = detect_file_extension(file_bytes) or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        source = tmp_path
        print(f"[Docling Worker] 📄 Archivo listo en {tmp_path} ({len(file_bytes) / 1024:.1f} KB). Iniciando Docling...", flush=True)

        # 1. Procesar con Docling (con auto-fallback a CPU si la GPU arroja error de kernel/arquitectura)
        try:
            result = converter.convert(source)
            doc = result.document
        except Exception as conv_err:
            err_str = str(conv_err)
            print(f"[Docling Worker] ⚠️ Advertencia en conversión GPU: {err_str}", flush=True)
            if "CUDA" in err_str or "kernel image" in err_str or "AcceleratorError" in err_str:
                print(f"[Docling Worker] ⚠️ Ejecutando auto-fallback en CPU...", flush=True)
                result = cpu_converter.convert(source)
                doc = result.document
            else:
                raise conv_err

        # 2. Corregir el árbol del documento Docling (fusionar tildes/acentos separados por EasyOCR)
        sanitize_docling_document(doc)

        # 3. Extraer páginas, elementos estructurados (tablas, filas, celdas, bloques) y tokens inmutables
        sorted_pages, tokens = extract_layout_pages(doc, tmp_path)

        # 4. Extracción de códigos QR (VeriFactu / TicketBAI)
        qr_codes = extract_qr_codes(doc, tmp_path)

        # 5. Limpieza y normalización de Markdown
        clean_md = clean_markdown_output(doc.export_to_markdown())

        print(f"[Docling Worker] ✅ Procesamiento exitoso: {len(sorted_pages)} páginas, {len(tokens)} tokens, {len(qr_codes)} QR(s)", flush=True)

        return {
            "status": "success",
            "markdown": clean_md,
            "text": clean_md,
            "qr_codes": qr_codes,
            "tokens": tokens,
            "pages": sorted_pages
        }

    except Exception as e:
        traceback.print_exc()
        print(f"[Docling Worker] ❌ Error procesando documento: {e}", flush=True)
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "failed"
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    print("[Docling Worker] 🚀 RunPod Serverless Worker iniciado y escuchando la cola de jobs...", flush=True)
    runpod.serverless.start({"handler": handler})
