"""
Detección y decodificación de códigos QR (VeriFactu, TicketBAI y URLs de pago) en PDFs e imágenes.
"""
import os
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

def extract_qr_codes(doc: Any, tmp_path: Optional[str]) -> List[Dict[str, Any]]:
    """Detecta y decodifica códigos QR (VeriFactu, TicketBAI, URLs de pago) en PDFs e imágenes."""
    qr_results = []
    detector = cv2.QRCodeDetector()

    def process_image(img_bgr, page_num=1):
        if img_bgr is None or img_bgr.size == 0:
            return
        try:
            success, decoded_info, points, _ = detector.detectAndDecodeMulti(img_bgr)
            if success and decoded_info:
                for text in decoded_info:
                    if text and text.strip():
                        val = text.strip()
                        if not any(q["value"] == val for q in qr_results):
                            is_verifactu = "agenciatributaria.gob.es" in val.lower() or "verifactu" in val.lower()
                            is_ticketbai = "tbai" in val.lower() or "ticketbai" in val.lower() or "gipuzkoa.eus" in val.lower() or "bizkaia.eus" in val.lower() or "araba.eus" in val.lower()
                            
                            qr_results.append({
                                "value": val,
                                "page": page_num,
                                "is_verifactu": is_verifactu,
                                "is_ticketbai": is_ticketbai,
                                "type": "verifactu" if is_verifactu else ("ticketbai" if is_ticketbai else "qr_code")
                            })
            else:
                text, pts, _ = detector.detectAndDecode(img_bgr)
                if text and text.strip():
                    val = text.strip()
                    if not any(q["value"] == val for q in qr_results):
                        is_verifactu = "agenciatributaria.gob.es" in val.lower() or "verifactu" in val.lower()
                        is_ticketbai = "tbai" in val.lower() or "ticketbai" in val.lower() or "gipuzkoa.eus" in val.lower() or "bizkaia.eus" in val.lower() or "araba.eus" in val.lower()
                        
                        qr_results.append({
                            "value": val,
                            "page": page_num,
                            "is_verifactu": is_verifactu,
                            "is_ticketbai": is_ticketbai,
                            "type": "verifactu" if is_verifactu else ("ticketbai" if is_ticketbai else "qr_code")
                        })
        except Exception:
            pass

    # 1. Si es PDF, renderizar cada página con pypdfium2 para máxima precisión
    if tmp_path and tmp_path.lower().endswith(".pdf"):
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(tmp_path)
            for page_idx in range(len(pdf)):
                page = pdf[page_idx]
                bitmap = page.render(scale=2)
                pil_image = bitmap.to_pil()
                img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                process_image(img_np, page_num=page_idx + 1)
        except Exception as e:
            print(f"[Docling Worker] Escaneo QR en PDF: {e}")

    # 2. Si es una imagen (PNG, JPG, WebP, etc.)
    elif tmp_path and os.path.exists(tmp_path):
        try:
            img = cv2.imread(tmp_path)
            process_image(img, page_num=1)
        except Exception as e:
            print(f"[Docling Worker] Escaneo QR en imagen: {e}")

    return qr_results

