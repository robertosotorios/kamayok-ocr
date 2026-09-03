"""
Configuración del pipeline de Docling, aceleración por GPU y conversión de documentos.
"""
import torch
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
    EasyOcrOptions,
)
from docling.datamodel.base_models import InputFormat

# 0. Diagnóstico y verificación de GPU
cuda_available = torch.cuda.is_available()
device = AcceleratorDevice.CUDA if cuda_available else AcceleratorDevice.CPU

print(f"[Docling Worker] CUDA Available: {cuda_available}")
if cuda_available:
    cap = torch.cuda.get_device_capability(0)
    print(f"[Docling Worker] GPU Model: {torch.cuda.get_device_name(0)} (Compute Capability: {cap[0]}.{cap[1]})")
    print(f"[Docling Worker] GPU Count: {torch.cuda.device_count()}")
    print(f"[Docling Worker] VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
    if arch_list:
        print(f"[Docling Worker] Supported CUDA Archs: {', '.join(arch_list)}")
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
pipeline_options.images_scale = 2.0

# Forzar el motor de OCR a procesar páginas completas (evita saltarse texto escaneado/imágenes)
if cuda_available:
    try:
        # Verificación activa de ejecución de kernel en GPU (soporta Ampere, Ada, Hopper y Blackwell)
        test_t = torch.zeros((1,), device="cuda")
        _ = test_t + 1
        pipeline_options.ocr_options = EasyOcrOptions(use_gpu=True, lang=["es", "en"], force_full_page_ocr=True)
        print(f"[Docling Worker] ✅ EasyOCR configurado con GPU activa ({torch.cuda.get_device_name(0)})")
    except Exception as e:
        print(f"[Docling Worker] ⚠️ Advertencia en inicialización GPU ({e}). Activando CPU fallback.")
        pipeline_options.ocr_options = EasyOcrOptions(use_gpu=False, lang=["es", "en"], force_full_page_ocr=True)
else:
    pipeline_options.ocr_options = EasyOcrOptions(use_gpu=False, lang=["es", "en"], force_full_page_ocr=True)

# 2. Inicialización de conversores soportando PDF y formatos de imagen (incluyendo WebP)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options)
    }
)

