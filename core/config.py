"""
Configuración del pipeline de Docling, aceleración por GPU y conversión de documentos.
"""
import os
import torch
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
    EasyOcrOptions,
    TableStructureOptions,
    TableFormerMode,
)
try:
    from docling.datamodel.pipeline_options import RapidOcrOptions
    has_rapid_ocr = True
except ImportError:
    has_rapid_ocr = False
from docling.datamodel.base_models import InputFormat

# 0. Diagnóstico y verificación de GPU
cuda_available = torch.cuda.is_available()
device = AcceleratorDevice.CUDA if cuda_available else AcceleratorDevice.CPU

print(f"[Docling Worker] CUDA Available: {cuda_available}", flush=True)
if cuda_available:
    cap = torch.cuda.get_device_capability(0)
    print(f"[Docling Worker] GPU Model: {torch.cuda.get_device_name(0)} (Compute Capability: {cap[0]}.{cap[1]})", flush=True)
    print(f"[Docling Worker] GPU Count: {torch.cuda.device_count()}", flush=True)
    print(f"[Docling Worker] VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)
    arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
    if arch_list:
        print(f"[Docling Worker] Supported CUDA Archs: {', '.join(arch_list)}", flush=True)
else:
    print("[Docling Worker] ⚠️ ADVERTENCIA: CUDA no detectado. Ejecutando en CPU fallback.", flush=True)

# Resolución de renderizado: 72 DPI * 3.5 ≈ 252 DPI, 4.167 ≈ 300 DPI
IMAGES_SCALE = float(os.environ.get("DOCLING_IMAGES_SCALE", "3.5"))
OCR_ENGINE_NAME = os.environ.get("OCR_ENGINE", "rapidocr").lower()

def create_ocr_options(use_gpu: bool = False):
    """Crea las opciones de OCR utilizando RapidOCR (preferente para facturas españolas) o EasyOCR."""
    if OCR_ENGINE_NAME == "rapidocr" and has_rapid_ocr:
        try:
            print(f"[Docling Worker] 🚀 Inicializando RapidOCR (force_full_page_ocr=True)...", flush=True)
            return RapidOcrOptions(force_full_page_ocr=True)
        except Exception as e:
            print(f"[Docling Worker] ⚠️ Error inicializando RapidOCR ({e}), fallback a EasyOCR.", flush=True)
    
    print(f"[Docling Worker] ℹ️ Inicializando EasyOCR (use_gpu={use_gpu}, force_full_page_ocr=True)...", flush=True)
    return EasyOcrOptions(use_gpu=use_gpu, lang=["es", "en"], force_full_page_ocr=True)

# 1. Configuración de Pipeline con aceleración por GPU
gpu_pipeline_options = PdfPipelineOptions()
gpu_pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4,
    device=AcceleratorDevice.CUDA if cuda_available else AcceleratorDevice.CPU
)
gpu_pipeline_options.do_table_structure = True
gpu_pipeline_options.table_structure_options = TableStructureOptions(
    mode=TableFormerMode.ACCURATE,
    do_cell_matching=True
)
gpu_pipeline_options.do_ocr = True
gpu_pipeline_options.images_scale = IMAGES_SCALE
gpu_pipeline_options.generate_page_images = True
gpu_pipeline_options.generate_picture_images = True

if cuda_available:
    try:
        # Verificación activa de ejecución de kernel en GPU
        test_t = torch.zeros((1,), device="cuda")
        _ = test_t + 1
        gpu_pipeline_options.ocr_options = create_ocr_options(use_gpu=True)
        print(f"[Docling Worker] ✅ OCR configurado con GPU activa ({torch.cuda.get_device_name(0)})", flush=True)
    except Exception as e:
        print(f"[Docling Worker] ⚠️ Advertencia en inicialización GPU ({e}). Activando CPU fallback.", flush=True)
        gpu_pipeline_options.ocr_options = create_ocr_options(use_gpu=False)
else:
    gpu_pipeline_options.ocr_options = create_ocr_options(use_gpu=False)

# 2. Configuración de Pipeline CPU seguro (Fallback garantizado)
cpu_pipeline_options = PdfPipelineOptions()
cpu_pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4,
    device=AcceleratorDevice.CPU
)
cpu_pipeline_options.do_table_structure = True
cpu_pipeline_options.table_structure_options = TableStructureOptions(
    mode=TableFormerMode.ACCURATE,
    do_cell_matching=True
)
cpu_pipeline_options.do_ocr = True
cpu_pipeline_options.images_scale = IMAGES_SCALE
cpu_pipeline_options.generate_page_images = True
cpu_pipeline_options.generate_picture_images = True
cpu_pipeline_options.ocr_options = create_ocr_options(use_gpu=False)

# 3. Inicialización de conversores
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=gpu_pipeline_options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=gpu_pipeline_options)
    }
)

cpu_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=cpu_pipeline_options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=cpu_pipeline_options)
    }
)

