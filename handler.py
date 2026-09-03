"""
Servicio OCR Docling + EasyOCR para RunPod Serverless y Load Balancers HTTP.
"""
import os
import base64
import tempfile
import torch
import runpod
from fastapi import FastAPI, Request
import uvicorn

from core.config import converter, cpu_converter, cuda_available
from core.geometry import detect_file_extension
from core.sanitizer import sanitize_docling_document, clean_markdown_output
from core.extractors import extract_layout_pages
from core.qr_scanner import extract_qr_codes

def handler(event):
    """Manejador principal de peticiones OCR."""
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

        # 1. Procesar con Docling (con auto-fallback a CPU si la GPU arroja error de kernel/arquitectura)
        try:
            result = converter.convert(source)
            doc = result.document
        except Exception as conv_err:
            err_str = str(conv_err)
            if "CUDA" in err_str or "kernel image" in err_str or "AcceleratorError" in err_str:
                print(f"[Docling Worker] ⚠️ Fallo en kernel CUDA ({conv_err}). Ejecutando auto-fallback en CPU...")
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

        return {
            "status": "success",
            "markdown": clean_md,
            "qr_codes": qr_codes,
            "tokens": tokens,
            "pages": sorted_pages
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# =============================================================================
# Servidor FastAPI para RunPod Load Balancer (/ping, /runsync, /run)
# =============================================================================

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
