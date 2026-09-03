FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_XET_HIGH_PERFORMANCE=1

# Instalar librerías de sistema requeridas por OpenCV y el renderizado de PDFs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python (incluyendo motores de OCR y escáner de QR / VeriFactu)
RUN pip install --no-cache-dir runpod docling easyocr rapidocr-onnxruntime opencv-python-headless pypdfium2

# PRE-DESCARGA: Cachear pesos de Docling y modelos OCR (español e inglés) en /root/.cache
RUN python3 -c "import easyocr; easyocr.Reader(['es', 'en'], gpu=False, download_enabled=True); from docling.document_converter import DocumentConverter; DocumentConverter()"

# Copiar el script que atiende las peticiones de RunPod
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
