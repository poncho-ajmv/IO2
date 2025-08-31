# archivo: generar_qr.py
# pip install qrcode[pil]

import qrcode
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H
import argparse

NIVELES = {"L": ERROR_CORRECT_L, "M": ERROR_CORRECT_M, "Q": ERROR_CORRECT_Q, "H": ERROR_CORRECT_H}

def crear_qr(dato: str,
             archivo: str = "qr.png",
             nivel: str = "M",
             tam_celda: int = 10,
             borde: int = 4) -> str:
    qr = qrcode.QRCode(
        version=None,                     # autoajuste
        error_correction=NIVELES[nivel],  # L/M/Q/H
        box_size=tam_celda,               # tamaño de cada celda (px)
        border=borde                      # margen (celdas)
    )
    qr.add_data(dato)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(archivo)
    return archivo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera un código QR desde un texto o URL.")
    parser.add_argument("dato", nargs="?", default="https://github.com/poncho-ajmv/IO2",
                        help="Texto o URL a codificar (por defecto: repo IO2).")
    parser.add_argument("-o", "--output", default="IO2_qr.png", help="Nombre del archivo de salida (PNG).")
    parser.add_argument("--nivel", choices=list(NIVELES.keys()), default="M",
                        help="Nivel de corrección de errores: L, M, Q, H.")
    parser.add_argument("--tam", type=int, default=10, help="Tamaño de celda (px).")
    parser.add_argument("--borde", type=int, default=4, help="Borde en celdas.")
    args = parser.parse_args()

    salida = crear_qr(args.dato, args.output, args.nivel, args.tam, args.borde)
    print(f"QR guardado en: {salida}")
