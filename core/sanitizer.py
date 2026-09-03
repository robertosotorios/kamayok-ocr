"""
Saneamiento y normalización de textos, tildes y salidas de Markdown.
"""
import re
from typing import Any

def fix_split_accents(text: str) -> str:
    """Corrige la separación errónea de vocales con tilde y caracteres generada por EasyOCR."""
    if not text:
        return text
    # Une letras con vocal acentuada en medio: "Depuraci ó n" -> "Depuración", "R ú a" -> "Rúa", "b á sico" -> "básico"
    text = re.sub(r'([a-zA-ZáéíóúÁÉÍÓÚñÑüÜ])\s+([áéíóúÁÉÍÓÚ])\s+([a-zA-ZáéíóúÁÉÍÓÚñÑüÜ])', r'\1\2\3', text)
    # Une al final: "Depuraci ó" -> "Depuració"
    text = re.sub(r'([a-zA-ZáéíóúÁÉÍÓÚñÑüÜ])\s+([áéíóúÁÉÍÓÚ])(?!\w)', r'\1\2', text)
    # Une al inicio: "ó n" -> "ón", "Á rea" -> "Área"
    text = re.sub(r'(^|\s)([áéíóúÁÉÍÓÚ])\s+([a-zA-ZáéíóúÁÉÍÓÚñÑüÜ])', r'\1\2\3', text)
    # Une ordinales: "N º" -> "Nº", "N ª" -> "Nª"
    text = re.sub(r'([a-zA-Z])\s+([ºª°])', r'\1\2', text)
    return text

def sanitize_docling_document(doc: Any) -> None:
    """Recorre el árbol de objetos de Docling para corregir tildes y caracteres antes de la extracción."""
    if hasattr(doc, "tables") and doc.tables:
        for table in doc.tables:
            if hasattr(table, "data") and hasattr(table.data, "table_cells") and table.data.table_cells:
                for cell in table.data.table_cells:
                    if hasattr(cell, "text") and cell.text:
                        cell.text = fix_split_accents(cell.text)

    for item, _ in doc.iterate_items():
        if hasattr(item, "text") and item.text:
            item.text = fix_split_accents(item.text)

def clean_markdown_output(raw_md: str) -> str:
    """Elimina comentarios de imagen residuales y saltos de línea redundantes del Markdown de Docling."""
    clean_md = (
        raw_md
        .replace("<!-- image -->", "")
        .replace("<!-- 🖼️❌ Image not available. Please use `PdfPipelineOptions(generate_picture_images=True)` -->", "")
    )
    clean_md = re.sub(r'\n{3,}', '\n\n', clean_md).strip()
    return fix_split_accents(clean_md)

