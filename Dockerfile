FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0+PTX" \
    CUDA_FORCE_PTX_JIT=1

# Instalar librerías de sistema requeridas por OpenCV y el renderizado de PDFs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Instalar PyTorch con soporte nativo para CUDA 12.8 / Blackwell (sm_100, sm_120) y librerías OCR
RUN pip install --no-cache-dir --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir runpod docling easyocr rapidocr-onnxruntime opencv-python-headless pypdfium2

# PRE-DESCARGA: Cachear pesos de Docling y modelos OCR (español e inglés) en /root/.cache
RUN python3 -c "import easyocr; easyocr.Reader(['es', 'en'], gpu=False, download_enabled=True); from docling.document_converter import DocumentConverter; DocumentConverter()"

# Copiar el código fuente y módulos del microservicio
COPY core/ /core/
COPY handler.py /handler.py

EXPOSE 80 8000

CMD ["python3", "-u", "/handler.py"]
