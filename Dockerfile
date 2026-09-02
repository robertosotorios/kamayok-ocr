FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Instalar librerías de sistema requeridas por OpenCV y el renderizado de PDFs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
RUN pip install --no-cache-dir runpod docling

# PRE-DESCARGA: Ejecutar inicialización en build para cachear pesos en /root/.cache
RUN python3 -c "from docling.document_converter import DocumentConverter; DocumentConverter()"

# Copiar el script que atiende las peticiones de RunPod
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
