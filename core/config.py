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

# 1. Configuración de Pipeline con aceleración por GPU
gpu_pipeline_options = PdfPipelineOptions()
gpu_pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4,
    device=AcceleratorDevice.CUDA if cuda_available else AcceleratorDevice.CPU
)
gpu_pipeline_options.do_table_structure = True
gpu_pipeline_options.do_ocr = True
gpu_pipeline_options.images_scale = 2.0

if cuda_available:
    try:
        # Verificación activa de ejecución de kernel en GPU
        test_t = torch.zeros((1,), device="cuda")
        _ = test_t + 1
        gpu_pipeline_options.ocr_options = EasyOcrOptions(use_gpu=True, lang=["es", "en"], force_full_page_ocr=True)
        print(f"[Docling Worker] ✅ EasyOCR configurado con GPU activa ({torch.cuda.get_device_name(0)})")
    except Exception as e:
        print(f"[Docling Worker] ⚠️ Advertencia en inicialización GPU ({e}). Activando CPU fallback.")
        gpu_pipeline_options.ocr_options = EasyOcrOptions(use_gpu=False, lang=["es", "en"], force_full_page_ocr=True)
else:
    gpu_pipeline_options.ocr_options = EasyOcrOptions(use_gpu=False, lang=["es", "en"], force_full_page_ocr=True)

# 2. Configuración de Pipeline CPU seguro (Fallback garantizado)
cpu_pipeline_options = PdfPipelineOptions()
cpu_pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4,
    device=AcceleratorDevice.CPU
)
cpu_pipeline_options.do_table_structure = True
cpu_pipeline_options.do_ocr = True
cpu_pipeline_options.images_scale = 2.0
cpu_pipeline_options.ocr_options = EasyOcrOptions(use_gpu=False, lang=["es", "en"], force_full_page_ocr=True)

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

