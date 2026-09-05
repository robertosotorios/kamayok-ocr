"""
Servicio OCR Docling + EasyOCR para RunPod Serverless y pruebas locales/HTTP.
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
from core.margin_recovery import recover_missing_margin_content

def handler(event):
    """
    Manejador principal de peticiones OCR.
    Compatible tanto con llamadas por cola (/run, /runsync) como directas por HTTP.
    """
    job_input = event.get("input", {}) if isinstance(event, dict) else {}
    if not job_input and isinstance(event, dict) and ("file_base64" in event or "file_url" in event):
        job_input = event
    
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

        # Preprocesamiento: rasterizado a 300 DPI si se activa force_rasterize o FORCE_RASTERIZE=true (por defecto True)
        rasterized_tmp = None
        force_rast = job_input.get("force_rasterize", True) or os.environ.get("FORCE_RASTERIZE", "true").lower() in ("1", "true", "yes")
        if ext == ".pdf" and force_rast:
            try:
                import pypdfium2 as pdfium
                pdf_doc = pdfium.PdfDocument(tmp_path)
                if len(pdf_doc) == 1:
                    print("[Docling Worker] 🖼️ Rasterizando PDF a PNG de 300 DPI antes de conversión...", flush=True)
                    page = pdf_doc[0]
                    bitmap = page.render(scale=4.166667)
                    pil_image = bitmap.to_pil()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as r_tmp:
                        pil_image.save(r_tmp.name, format="PNG")
                        rasterized_tmp = r_tmp.name
                        source = rasterized_tmp
            except Exception as r_err:
                print(f"[Docling Worker] ⚠️ Advertencia en rasterizado previo: {r_err}", flush=True)

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
        sorted_pages, tokens = extract_layout_pages(doc, source)

        # 4. Extracción de códigos QR (VeriFactu / TicketBAI)
        qr_codes = extract_qr_codes(doc, source)

        # 5. Limpieza y normalización de Markdown (con traverse_pictures=True para incluir texto en cajas gráficas)
        try:
            raw_md = doc.export_to_markdown(traverse_pictures=True)
        except TypeError:
            raw_md = doc.export_to_markdown()
        clean_md = clean_markdown_output(raw_md)

        # 6. Recuperación de textos marginales y verticales (CIFs, datos registrales y notas legales en bordes)
        id_counter = {"token": len(tokens), "line": len(tokens), "block": 500}
        clean_md, sorted_pages, tokens = recover_missing_margin_content(
            sorted_pages, tokens, clean_md, source, id_counter
        )

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
        for p in (tmp_path, rasterized_tmp):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


if __name__ == "__main__":
    # Si se pasa el flag --http o la variable RUNPOD_HTTP_MODE=true se activa FastAPI para pruebas locales o HTTP directos
    if "--http" in sys.argv or os.environ.get("RUNPOD_HTTP_MODE", "").lower() in ("1", "true", "yes"):
        import uvicorn
        from fastapi import FastAPI, Request
        
        app = FastAPI(title="Kamayok OCR Docling Service")

        @app.get("/ping")
        @app.get("/health")
        def ping():
            return {
                "status": "healthy",
                "cuda_available": cuda_available,
                "gpu_model": torch.cuda.get_device_name(0) if cuda_available else "CPU"
            }

        @app.post("/runsync")
        @app.post("/run")
        @app.post("/")
        async def direct_endpoint(request: Request):
            try:
                body = await request.json()
            except Exception:
                body = {}
            event = body if "input" in body else {"input": body}
            result = handler(event)
            return {
                "status": "COMPLETED",
                "output": result
            }

        port = int(os.environ.get("PORT", "8000"))
        print(f"[Docling Worker] 🌐 Servidor HTTP FastAPI activo en puerto {port}...", flush=True)
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # Modo RunPod Serverless estándar: RunPod gestiona automáticamente colas (/run), webhooks y llamadas síncronas (/runsync)
        print("[Docling Worker] 🚀 RunPod Serverless Worker activo (Soporta /run con Webhook y /runsync síncrono nativamente)...", flush=True)
        runpod.serverless.start({"handler": handler})
