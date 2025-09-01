import argparse
from pathlib import Path
import time
import sys
import threading
import queue

import cv2
import numpy as np

# ============================
# Configuración rápida
# ============================
YOLO_INTERVAL = 2     # corre YOLO cada N frames (2 o 3 suele ir bien)
POSE_IMGSZ   = 512    # tamaño de inferencia (512 acelera vs 640)
USE_TRT      = False  # si ya exportaste yolov8n-pose.engine, pon True

# ============================
# CUDA / Torch (GPU)
# ============================
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
except Exception:
    HAS_CUDA = False

# ============================
# YOLOv8 (Ultralytics)
# ============================
HAS_YOLO = True
try:
    from ultralytics import YOLO
except Exception:
    HAS_YOLO = False
    YOLO = None

# ============================
# OpenCV CUDA (opcional)
# ============================

def _cv_cuda_available() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

HAS_CV_CUDA = _cv_cuda_available()
if HAS_CV_CUDA:
    try:
        cv2.cuda.setDevice(0)
    except Exception:
        pass

# ============================
# Utilidades de dibujo
# ============================

def draw_text(img, text, org=(10, 30), scale=0.8, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# ============================
# Convoluciones (presets + processor)
# ============================

CONV_PRESETS = [
    ("Identidad", {"type": "kernel", "kernel": [[0,0,0],[0,1,0],[0,0,0]], "div": 1}),
    ("Eje basico", {"type": "kernel", "kernel": [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], "div": 1}),
    ("Desenfoque basico", {"type": "kernel", "kernel": [[1,1,1],[1,1,1],[1,1,1]], "div": 9}),
    ("Desenfoque Gaussiano", {"type": "kernel", "kernel": [
        [1, 4, 6, 4, 1],
        [4,16,24,16, 4],
        [6,24,36,24, 6],
        [4,16,24,16, 4],
        [1, 4, 6, 4, 1],
    ], "div": 256}),
    ("Enfocar", {"type": "kernel", "kernel": [[0,-1,0],[-1,5,-1],[0,-1,0]], "div": 1}),
    ("Realzar", {"type": "kernel", "kernel": [[-2,-1,0],[-1,1,1],[0,1,2]], "div": 1}),
    ("Sobel horizontal", {"type": "kernel", "kernel": [[-1,-2,-1],[0,0,0],[1,2,1]], "div": 1}),
    ("Sobel vertical",   {"type": "kernel", "kernel": [[-1,0,1],[-2,0,2],[-1,0,1]], "div": 1}),
    ("Sobel HyV", {"type": "sobel", "blur": False, "colorize": False, "th": None}),
    ("Sobel HyV con Blur", {"type": "sobel", "blur": True, "colorize": False, "th": None}),
    ("Sobel HyV con Blur y TH", {"type": "sobel", "blur": True, "colorize": False, "th": 40}),
    ("Sobel colorizado", {"type": "sobel", "blur": False, "colorize": True, "th": None}),
    ("Sobel colorizado con Blur", {"type": "sobel", "blur": True, "colorize": True, "th": None}),
]

class ConvolutionProcessor:
    def __init__(self):
        self.use_gpu = HAS_CV_CUDA
        if self.use_gpu:
            try:
                self._gpu_bgr = cv2.cuda_GpuMat()
            except Exception:
                self.use_gpu = False

    @staticmethod
    def _kernel_to_np(kernel, divisor=1):
        d = float(divisor) if divisor else 1.0
        return (np.array(kernel, dtype=np.float32) / d)

    def apply_kernel(self, bgr, kernel, divisor=1):
        k = self._kernel_to_np(kernel, divisor)
        return cv2.filter2D(bgr, ddepth=-1, kernel=k, borderType=cv2.BORDER_DEFAULT)

    def apply_sobel(self, bgr, blur=False, colorize=False, th=None):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if blur:
            gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        if th is not None and th > 0:
            mag = np.where(mag < th, 0, mag).astype(np.float32)
        mag_max = float(mag.max()) if mag.size else 1.0
        if mag_max < 1e-6:
            mag_max = 1.0
        if colorize:
            ang = cv2.phase(gx, gy, angleInDegrees=True)
            h = np.uint8(np.clip(ang / 2.0, 0, 179))
            s = np.full_like(h, 255, dtype=np.uint8)
            v = np.uint8(np.clip((mag / mag_max) * 255.0, 0, 255))
            hsv = cv2.merge([h, s, v])
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            v = np.uint8(np.clip((mag / mag_max) * 255.0, 0, 255))
            out = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
        return out

    def apply(self, bgr, preset_cfg):
        if preset_cfg["type"] == "kernel":
            return self.apply_kernel(bgr, preset_cfg["kernel"], preset_cfg.get("div", 1))
        else:
            return self.apply_sobel(
                bgr,
                blur=preset_cfg.get("blur", False),
                colorize=preset_cfg.get("colorize", False),
                th=preset_cfg.get("th", None),
            )

# ============================
# Edge processor (GPU si hay OpenCV CUDA)
# ============================
class EdgeProcessor:
    def __init__(self, canny_th1=80, canny_th2=160, blur_ks=(5, 5), blur_sigma=1.2):
        self.use_gpu = HAS_CV_CUDA
        self.canny_th1 = canny_th1
        self.canny_th2 = canny_th2
        self.blur_ks = blur_ks
        self.blur_sigma = blur_sigma
        if self.use_gpu:
            self._gf = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, self.blur_ks, self.blur_sigma)
            self._canny = cv2.cuda.createCannyEdgeDetector(self.canny_th1, self.canny_th2)
            self._gpu_bgr = cv2.cuda_GpuMat()
            self._gpu_gray = cv2.cuda_GpuMat()
            self._gpu_blur = cv2.cuda_GpuMat()
            self._gpu_edges = cv2.cuda_GpuMat()

    def process(self, bgr, min_area=500):
        if self.use_gpu:
            self._gpu_bgr.upload(bgr)
            self._gpu_gray = cv2.cuda.cvtColor(self._gpu_bgr, cv2.COLOR_BGR2GRAY)
            self._gpu_blur = self._gf.apply(self._gpu_gray)
            self._gpu_edges = self._canny.detect(self._gpu_blur)
            edges = self._gpu_edges.download()
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, self.blur_ks, self.blur_sigma)
            edges = cv2.Canny(blur, self.canny_th1, self.canny_th2)

        annotated = bgr.copy()
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            m = cv2.moments(c)
            if m["m00"] != 0:
                cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(annotated, 0.8, edges_rgb, 0.7, 0)
        return overlay, edges

# ============================
# Fallback CPU pipeline
# ============================

def process_image_pipeline(bgr, min_area=500):
    annotated = bgr.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blur, 80, 160)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        m = cv2.moments(c)
        if m["m00"] != 0:
            cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(annotated, 0.8, edges_rgb, 0.7, 0)
    return overlay, edges


def face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("No pude cargar el clasificador Haar: " + cascade_path)
    return face_cascade

# ============================
# YOLOv8-Pose -> manos (ORIGINAL) — ahora “mirror-aware”
# ============================
class PoseHandEstimator:
    """
    Usa yolov8n-pose (COCO 17 kpts) y dibuja muñeca + linea codo→muñeca + caja en la muñeca.
    Si mirrored=True, invierte la etiqueta Izq/Der para que coincida con la vista espejada.
    """
    def __init__(self, imgsz=640, conf=0.5, use_trt=False, warmup=True, mirrored=True):
        if not HAS_YOLO:
            raise RuntimeError("Ultralytics YOLO no está disponible")
        model_path = "yolov8n-pose.engine" if use_trt else "yolov8n-pose.pt"
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.device = 0 if HAS_CUDA else "cpu"
        self.fp16 = bool(HAS_CUDA)
        self.runtime = "TRT FP16" if use_trt and HAS_CUDA else ("GPU FP16" if HAS_CUDA else "CPU")
        self.mirrored = bool(mirrored)
        if warmup:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy, imgsz=self.imgsz, conf=self.conf, device=self.device, half=self.fp16, verbose=False
            )[0]

    @staticmethod
    def _to_np(x):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            return np.array(x)

    def infer(self, bgr):
        res = self.model.predict(
            bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, half=self.fp16, verbose=False
        )[0]
        if res.keypoints is None:
            return None, None
        kpts = self._to_np(res.keypoints.xy)  # [N,17,2]
        boxes = None
        if getattr(res, "boxes", None) is not None and getattr(res.boxes, "xyxy", None) is not None:
            boxes = self._to_np(res.boxes.xyxy).astype(int)
        return kpts, boxes

    def draw(self, canvas, kpts, boxes):
        if kpts is None:
            return canvas
        if boxes is not None:
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (60, 200, 255), 1)

        for i in range(kpts.shape[0]):
            left_elbow, left_wrist = kpts[i, 7], kpts[i, 9]
            right_elbow, right_wrist = kpts[i, 8], kpts[i, 10]
            for wrist, elbow, side in [(left_wrist, left_elbow, "Izq"), (right_wrist, right_elbow, "Der")]:
                wx, wy = wrist
                ex, ey = elbow
                if (wx <= 0 and wy <= 0) or (ex <= 0 and ey <= 0):
                    continue
                # Ajuste etiqueta por espejo
                side_lbl = {"Izq": "Der", "Der": "Izq"}[side] if self.mirrored else side

                d = np.linalg.norm([wx - ex, wy - ey])
                s = int(np.clip(d * 0.9, 40, 180))
                x0, y0 = int(wx - s / 2), int(wy - s / 2)
                x1, y1 = x0 + s, y0 + s
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 0, 180), 2)
                cv2.line(canvas, (int(ex), int(ey)), (int(wx), int(wy)), (255, 0, 180), 2)
                cv2.circle(canvas, (int(wx), int(wy)), 6, (255, 0, 180), -1)
                v = np.array([wx - ex, wy - ey], dtype=float)
                n = np.linalg.norm(v)
                if n > 1:
                    pv = np.array([-v[1], v[0]]) / n
                    p1 = (int(wx + 8 * pv[0]), int(wy + 8 * pv[1]))
                    p2 = (int(wx - 8 * pv[0]), int(wy - 8 * pv[1]))
                    cv2.line(canvas, p1, p2, (255, 0, 180), 2)
                draw_text(canvas, f"Mano {side_lbl}", (x0, max(0, y0 - 8)), 0.6, (255, 0, 180), 2)
        return canvas

# ============================
# YOLOv8-Hands (21 kpts) + RPS — MODO 8, con mejora para “Tijera”
# ============================

HAND_MODEL_PATH = "yolov8n-hand.pt"   # cambia si tu checkpoint está en otra ruta

class HandRPSDetector:
    """
    Detector de manos con 21 keypoints + clasificación Piedra/Papel/Tijera mejorada.
    - Umbrales orient.-invariantes (longitud de dedo y ángulo).
    - Deducción de diestro/zurdo y ajuste por imagen espejada.
    """
    def __init__(self, model_path=HAND_MODEL_PATH, imgsz=640, conf=0.5, iou=0.45,
                 max_hands=2, warmup=True, mirrored=True):
        if not HAS_YOLO:
            raise RuntimeError("Ultralytics YOLO no está disponible")
        self.model = YOLO(model_path)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou  = float(iou)
        self.max_hands = int(max_hands)
        self.device = 0 if HAS_CUDA else "cpu"
        self.fp16 = bool(HAS_CUDA)
        self.runtime = "GPU FP16" if HAS_CUDA else "CPU"
        self.mirrored = bool(mirrored)

        # Índices tipo MediaPipe
        self.WRIST = 0
        self.FINGERS = {
            "thumb":  (1, 2, 3, 4),     # CMC, MCP, IP, TIP
            "index":  (5, 6, 7, 8),     # MCP, PIP, DIP, TIP
            "middle": (9,10,11,12),
            "ring":   (13,14,15,16),
            "pinky":  (17,18,19,20),
        }

        from collections import deque
        self._buffer = deque(maxlen=7)
        self._colors = {
            "Piedra": (0, 140, 255),
            "Papel": (0, 255, 0),
            "Tijera": (255, 0, 255),
            "Desconocido": (0, 0, 255),
        }
        if warmup:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model.predict(dummy, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                   device=self.device, half=self.fp16, verbose=False)[0]

    @staticmethod
    def _to_np(x):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            return np.array(x)

    def infer(self, bgr):
        res = self.model.predict(
            bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            device=self.device, half=self.fp16, verbose=False
        )[0]
        if res.keypoints is None or res.keypoints.xy is None:
            return None, None
        kpts = self._to_np(res.keypoints.xy)   # [N,21,2]
        if kpts.ndim != 3 or kpts.shape[1] < 21:
            return None, None

        boxes = None
        if getattr(res, "boxes", None) is not None and getattr(res.boxes, "xyxy", None) is not None:
            boxes = self._to_np(res.boxes.xyxy).astype(int)
            if getattr(res.boxes, "conf", None) is not None:
                order = np.argsort(-self._to_np(res.boxes.conf).reshape(-1))
                boxes = boxes[order][:self.max_hands]
                kpts  = kpts[order][:self.max_hands]
            else:
                boxes = boxes[:self.max_hands]
                kpts  = kpts[:self.max_hands]
        else:
            kpts = kpts[:self.max_hands]

        return kpts, boxes

    # ---------- utilidades de geometría normalizada ----------
    def _bbox_from_xy(self, xy):
        xs = xy[:,0]; ys = xy[:,1]
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _dims_from(self, xy, bbox=None):
        if bbox is not None and len(bbox)==4:
            x1,y1,x2,y2 = bbox
            W = max(1, x2-x1); H = max(1, y2-y1)
        else:
            x1,y1,x2,y2 = self._bbox_from_xy(xy)
            W = max(1, x2-x1); H = max(1, y2-y1)
        return W, H

    def _is_right_hand(self, xy):
        """
        Estima si es mano derecha mirando MCP índice (5) vs MCP meñique (17):
        - No espejado: x5 < x17  => derecha
        - Espejado: invertir la condición
        """
        x5 = xy[self.FINGERS["index"][0], 0]
        x17= xy[self.FINGERS["pinky"][0], 0]
        is_right = x5 < x17
        if self.mirrored:
            is_right = not is_right
        return is_right

    def _extended_finger(self, xy, name, H, alpha_up=0.035, alpha_tol=0.015):
        """
        (Versión original por Y) Dedo extendido si TIP claramente por encima (menor y) de PIP.
        """
        mcp, pip, dip, tip = self.FINGERS[name]
        tip_y = xy[tip, 1]; pip_y = xy[pip, 1]
        margin = (pip_y - tip_y) / max(1.0, H)  # >0 si tip está arriba
        return (margin > alpha_up), margin

    # ======= NUEVO: eje de la mano (wrist -> palma) =======
    def _hand_axis(self, xy):
        wrist = xy[self.WRIST]
        mid_mcp = xy[self.FINGERS["middle"][0]]
        idx_mcp = xy[self.FINGERS["index"][0]]
        v = (mid_mcp - wrist) + (idx_mcp - wrist)  # promedio robusto
        n = np.linalg.norm(v)
        if n < 1e-6:
            return np.array([0.0, -1.0], dtype=float)
        return v / n  # unitario

    # ======= NUEVO: dedo extendido orient.-invariante =======
    def _extended_finger_oriented(self, xy, name, H, alpha_proj=0.06, ang_thr_deg=35):
        mcp, pip, _, tip = self.FINGERS[name]
        u = self._hand_axis(xy)                     # “arriba” de la mano
        v_tip = xy[tip] - xy[mcp]
        v_pip = xy[pip] - xy[mcp]
        # proyección a lo largo del eje de la mano
        proj_tip = float(np.dot(v_tip, u)) / max(1.0, H)
        proj_pip = float(np.dot(v_pip, u)) / max(1.0, H)
        margin = proj_tip - proj_pip               # >0 si la punta está más “arriba”
        # ángulo del dedo respecto al eje de la mano (para evitar diagonales)
        den = max(1e-6, np.linalg.norm(v_tip))
        cosang = float(np.clip(np.dot(v_tip, u) / den, -1.0, 1.0))
        ang = np.degrees(np.arccos(cosang))
        return (margin > alpha_proj and ang < ang_thr_deg), margin

    def _extended_thumb(self, xy, W, alpha_side=0.035):
        """
        Pulgar: usa desplazamiento horizontal TIP–IP según diestro/zurdo.
        """
        _, mcp, ip, tip = self.FINGERS["thumb"]
        tip_x = xy[tip, 0]; ip_x = xy[ip, 0]
        is_right = self._is_right_hand(xy)
        if is_right:
            return ( (tip_x - ip_x) / max(1.0, W) > alpha_side )
        else:
            return ( (ip_x - tip_x) / max(1.0, W) > alpha_side )

    def _palm_center_and_radius(self, xy):
        """
        Centro y radio de la palma a partir de WRIST + MCPs (5,9,13,17).
        Radio = media de distancias al centro -> sirve como escala invariante.
        """
        wrist  = xy[self.WRIST]
        mcp_i  = xy[self.FINGERS["index"][0]]
        mcp_m  = xy[self.FINGERS["middle"][0]]
        mcp_r  = xy[self.FINGERS["ring"][0]]
        mcp_p  = xy[self.FINGERS["pinky"][0]]
        P = np.stack([wrist, mcp_i, mcp_m, mcp_r, mcp_p], axis=0)
        C = P.mean(axis=0)
        R = float(np.mean(np.linalg.norm(P - C, axis=1)))
        return C, max(1.0, R)

    def _ring_pinky_near_center(self, xy, thr=0.90):
        """
        True si tips de anular y meñique están cerca del centro de la palma.
        thr ~0.85–0.95 (más alto = más permisivo).
        """
        C, R = self._palm_center_and_radius(xy)
        ring_tip  = xy[self.FINGERS["ring"][3]]   # 16
        pinky_tip = xy[self.FINGERS["pinky"][3]]  # 20
        dr = np.linalg.norm(ring_tip  - C) / R
        dp = np.linalg.norm(pinky_tip - C) / R
        return (dr <= thr) and (dp <= thr)

    def _clasificar(self, xy, bbox=None):
        """
        Clasifica SIEMPRE en {Piedra, Papel, Tijera}.
        - Reglas duras para Piedra y Papel.
        - Tijera prioritaria cuando: solo hay dos dedos (índice+medio), o
        anular/meñique están cerca de la palma, o la "V" no es muy abierta.
        - Métricas invar. a orientación: separación normalizada por longitud de dedo + ángulo.
        """
        W, H = self._dims_from(xy, bbox)

        # ===== Extensión orient.-invariante (bool, margen) =====
        ext_index, m_idx  = self._extended_finger_oriented(xy, "index",  H)
        ext_middle, m_mid = self._extended_finger_oriented(xy, "middle", H)
        ext_ring,  m_rin  = self._extended_finger_oriented(xy, "ring",   H)
        ext_pinky, m_pin  = self._extended_finger_oriented(xy, "pinky",  H)

        count_up4 = int(ext_index) + int(ext_middle) + int(ext_ring) + int(ext_pinky)

        # Tolerancias para márgenes
        tol_up   = 0.02
        tol_down = 0.01
        # Papel más exigente en márgenes de "arriba"
        tol_up_papel = 0.050  # antes 0.035

        # ===== Índice vs Medio: separación (por longitud de dedo) + ángulo =====
        i_mcp = xy[self.FINGERS["index"][0]]
        m_mcp = xy[self.FINGERS["middle"][0]]
        i_tip = xy[self.FINGERS["index"][3]]
        m_tip = xy[self.FINGERS["middle"][3]]

        Li = np.linalg.norm(i_tip - i_mcp)
        Lm = np.linalg.norm(m_tip - m_mcp)
        L  = max(1.0, (Li + Lm) * 0.5)

        sep = np.linalg.norm(i_tip - m_tip) / L
        vi, vm = (i_tip - i_mcp), (m_tip - m_mcp)
        den = max(1e-6, np.linalg.norm(vi) * np.linalg.norm(vm))
        cosang  = float(np.clip(np.dot(vi, vm) / den, -1.0, 1.0))
        ang_deg = float(np.degrees(np.arccos(cosang)))

        # ===== Umbrales (Tijera más permisiva / Papel más exigente) =====
        SEP_TIJERA = 0.72   # antes 0.62 (V más abierta aún cuenta como Tijera)
        SEP_PAPEL  = 0.95   # antes 0.90 (Papel requiere apertura mayor)
        ANG_TIJERA = 42.0   # antes 36.0 (más fácil ser Tijera)
        ANG_PAPEL  = 68.0   # antes 60.0 (Papel requiere ángulo mayor)

        # “Arriba” usando el eje de la mano
        u = self._hand_axis(xy)
        up_idx = (np.dot(i_tip - xy[self.WRIST], u) / max(1.0, H)) > 0.02
        up_mid = (np.dot(m_tip - xy[self.WRIST], u) / max(1.0, H)) > 0.02

        # ===== Anular+meñique cerca del centro de la palma => favorecer TIJERA =====
        rp_thr = 0.965  # antes 0.94 (más permisivo para Tijera)
        rp_near = self._ring_pinky_near_center(xy, thr=rp_thr)
        if rp_near and (ext_index or m_idx > tol_up) and (ext_middle or m_mid > tol_up):
            return "Tijera"

        # ===== PAPEL (regla dura: 4 dedos bien extendidos y mano realmente abierta) =====
        if (ext_index and ext_middle and ext_ring and ext_pinky and
            (m_idx > tol_up_papel) and (m_mid > tol_up_papel) and
            (m_rin > tol_up_papel) and (m_pin > tol_up_papel) and
            (sep >= SEP_PAPEL or ang_deg >= ANG_PAPEL)):
            return "Papel"

        # ===== PIEDRA (regla dura) =====
        if (not ext_index and m_idx < tol_down) and \
        (not ext_middle and m_mid < tol_down) and \
        (not ext_ring  and m_rin < tol_down) and \
        (not ext_pinky and m_pin < tol_down):
            return "Piedra"

        # ===== REGLA DURA: exactamente dos dedos (índice+medio) arriba => TIJERA =====
        if (count_up4 == 2) and ext_index and ext_middle and (not ext_ring) and (not ext_pinky):
            return "Tijera"

        # ===== TIJERA / PAPEL con 2 dedos arriba =====
        cond_up = (ext_index or m_idx > tol_up) and (ext_middle or m_mid > tol_up)
        cond_dn = (not ext_ring or m_rin < tol_down) and (not ext_pinky or m_pin < tol_down)
        if cond_up and cond_dn:
            # Más permisivo para Tijera
            if (sep <= SEP_TIJERA) or (ang_deg <= ANG_TIJERA):
                return "Tijera"

            # Para que sea PAPEL con solo dos dedos visibles:
            # Debe ser MUY abierto, ambos (índice y medio) claramente arriba
            # y que asomen ANULAR **y** MEÑIQUE (ambos).
            very_open = (sep >= SEP_PAPEL) and (ang_deg >= ANG_PAPEL) and (up_idx and up_mid)
            if very_open and (ext_ring and ext_pinky):
                return "Papel"

            # Resto: mantener TIJERA
            return "Tijera"

        # ===== TIJERA (relajación adicional y fallback) =====
        # (a) Si la V es moderada y alguno de los otros dos no está claramente extendido.
        if (cond_up and ((sep <= SEP_TIJERA*1.05) or (ang_deg <= ANG_TIJERA+4)) and
            (m_rin < tol_up_papel or m_pin < tol_up_papel)):
            return "Tijera"

        # (b) “dos dedos muy juntos”: al menos uno algo arriba para evitar puño.
        if (sep <= SEP_TIJERA) and (up_idx or up_mid) and (ext_index or ext_middle):
            return "Tijera"

        # ===== PUNTAJE DE SIMILITUD (si nada calza) =====
        def pos(x, b=0.015):  # aporte por estar extendido
            return max(0.0, x - b)
        def neg(x, b=0.02):   # aporte por estar doblado
            return max(0.0, b - x)

        # Penalizo más a Papel por "apertura" y premio más a Tijera por pareja
        bonus_spread_papel = max(0.0, sep - SEP_PAPEL) * 0.15   # antes 0.25
        bonus_pair_tijera  = max(0.0, SEP_TIJERA - sep) * 1.30  # antes 1.10

        score_papel  = pos(m_idx) + pos(m_mid) + pos(m_rin) + pos(m_pin) + bonus_spread_papel
        score_piedra = neg(m_idx) + neg(m_mid) + neg(m_rin) + neg(m_pin)
        score_tijera = pos(m_idx) + pos(m_mid) + neg(m_rin) + neg(m_pin) + bonus_pair_tijera

        scores = {"Piedra": score_piedra, "Papel": score_papel, "Tijera": score_tijera}
        return max(scores, key=scores.get)




    def draw(self, canvas, kpts, boxes):
        if kpts is None:
            return canvas
        for i in range(kpts.shape[0]):
            xy = kpts[i]
            bbox = None
            if boxes is not None and i < (boxes.shape[0] if isinstance(boxes, np.ndarray) else len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                bbox = (x1,y1,x2,y2)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 255, 60), 2)

            gesto = self._clasificar(xy, bbox=bbox)
            self._buffer.append(gesto)
            estable = max(set(self._buffer), key=self._buffer.count) if len(self._buffer) > 0 else gesto
            col = self._colors.get(estable, (255, 255, 255))

            # puntos 21kp
            for (px, py) in xy:
                cx, cy = int(px), int(py)
                cv2.circle(canvas, (cx, cy), 2, (0, 0, 0), cv2.FILLED)
                cv2.circle(canvas, (cx, cy), 2, (255, 255, 255), 1)

            # Etiqueta
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.putText(canvas, f"{estable}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
            else:
                x0, y0 = int(xy[0, 0]), int(xy[0, 1])
                cv2.putText(canvas, f"{estable}", (x0 + 8, max(0, y0 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        return canvas

# ============================
# Cámara: FrameReader con drop de frames viejos
# ============================
class FrameReader:
    """Captura en hilo separado y mantiene SOLO el frame más reciente."""
    def __init__(self, cam_index=0, w=960, h=540, fps=30, prefer_mjpg=True):
        self.cam_index = cam_index
        self.w, self.h, self.fps = w, h, fps
        self.prefer_mjpg = prefer_mjpg
        self.cap = open_camera(cam_index, w, h, prefer_mjpg=prefer_mjpg, target_fps=fps)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"No pude abrir la cámara {cam_index}")
        self.q = queue.Queue(maxsize=1)
        self.running = False
        self.misses = 0
        self.t = None

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None or frame.size == 0:
                self.misses += 1
                time.sleep(0.005)
                continue
            self.misses = 0
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                pass

    def start(self):
        self.running = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def read(self, timeout=0.1):
        try:
            frame = self.q.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            return False, None

    def reopen(self):
        try:
            self.cap.release()
        except Exception:
            pass
        self.cap = open_camera(self.cam_index, self.w, self.h, self.prefer_mjpg, self.fps)
        self.misses = 0

    def stop(self):
        self.running = False
        if self.t:
            self.t.join(timeout=0.5)
        try:
            self.cap.release()
        except Exception:
            pass

# ============================
# Cámara (helpers)
# ============================

def open_camera(index=0, w=960, h=540, prefer_mjpg=True, target_fps=30):
    def _try_open(backend):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        if prefer_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap if cap.isOpened() else None

    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        cap = _try_open(backend)
        if cap is not None:
            return cap

    cap = cv2.VideoCapture(index)
    if prefer_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap

# ============================
# Modo IMAGEN
# ============================

def run_image_demo(image_path: Path, save_out=True):
    if not image_path.exists():
        print(f"[ERROR] No encuentro la imagen: {image_path}")
        sys.exit(1)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print("[ERROR] No pude leer la imagen.")
        sys.exit(1)

    edges_proc = EdgeProcessor()
    overlay, edges = edges_proc.process(bgr)

    cv2.imshow("Imagen anotada (Canny + Contornos + Centroides)", overlay)
    cv2.imshow("Edges (Canny)", edges)
    if save_out:
        out_path = image_path.with_name(image_path.stem + "_annotated.jpg")
        cv2.imwrite(str(out_path), overlay)
        print(f"[OK] Guardado: {out_path}")
    print("Presiona 'q' para cerrar.")
    while True:
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cv2.destroyAllWindows()

# ============================
# HUD y loop de cámara
# ============================

def draw_hud(frame, fps, mode, has_yolo, yolo_runtime, pose_imgsz, yolo_interval, conv_name=None):
    y = 24
    draw_text(frame, f"FPS: {fps:.1f}", (10, y)); y += 24
    draw_text(frame, f"Modo {mode}:", (10, y)); y += 24
    texts = {
        1: "1) Base",
        2: "2) Edges (Canny)",
        3: "3) Caras (Haar)",
        4: "4) Manos (YOLOv8-Pose original, espejo corregido)",
        5: "5) Caras + Manos (Pose)",
        6: "6) Caras + Manos + Edges (Pose)",
        7: "7) Convoluciones ([ / ] para cambiar)",
        8: "8) Manos (21 kpts + RPS mejorado)",
    }
    draw_text(frame, texts.get(mode, ""), (10, y), 0.6); y += 20
    draw_text(frame, "Teclas: 1..8 | [ ] cambia filtro | s snapshot | q salir", (10, y), 0.6); y += 20
    if mode == 7 and conv_name:
        draw_text(frame, f"Filtro: {conv_name}", (10, y), 0.6, (200, 255, 120), 2); y += 20

    if not has_yolo:
        draw_text(frame, "Nota: instala 'ultralytics' para modos 4-6 y 8", (10, y), 0.6, (0, 0, 255), 2); y += 20
    y += 2
    draw_text(frame, f"YOLO: {yolo_runtime} | imgsz={pose_imgsz} | interval={yolo_interval}", (10, y), 0.6); y += 20
    draw_text(frame, f"Torch CUDA: {'ON' if HAS_CUDA else 'OFF'}", (10, y), 0.6); y += 20
    draw_text(frame, f"OpenCV CUDA (edges): {'ON' if HAS_CV_CUDA else 'OFF'}", (10, y), 0.6); y += 20


def run_camera_demo(cam_index=0):
    reader = FrameReader(cam_index).start()

    cv2.setUseOptimized(True)
    try:
        if HAS_CUDA and hasattr(cv2.ocl, "setUseOpenCL"):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    face_casc = face_detector()

    # Detectores
    yolo_pose = PoseHandEstimator(imgsz=POSE_IMGSZ, conf=0.5, use_trt=USE_TRT, mirrored=True) if HAS_YOLO else None
    yolo_hand21 = HandRPSDetector(
        model_path=HAND_MODEL_PATH, imgsz=POSE_IMGSZ, conf=0.5, iou=0.45, max_hands=2, warmup=True, mirrored=True
    ) if HAS_YOLO else None

    edges_proc = EdgeProcessor()
    conv_proc = ConvolutionProcessor()
    conv_idx = 0  # índice del filtro actual en CONV_PRESETS

    # Por defecto: si hay YOLO -> 4; si no -> 3
    mode = 4 if HAS_YOLO else 3

    t0 = time.time(); frames = 0; fps = 0.0
    print("Modos: 1=Base  2=Edges  3=Caras  4=Manos(Pose)  5=Caras+Manos(Pose)  6=Caras+Manos+Edges(Pose)  7=Convoluciones  8=Manos(21kpts+RPS)")
    print("Teclas: 1..8 | [ ] cambia filtro | s snapshot | q salir")
    print(f"[INFO] Torch CUDA: {HAS_CUDA} | OpenCV CUDA: {HAS_CV_CUDA} | Ultralytics: {HAS_YOLO}")
    if yolo_pose is not None:
        print(f"[INFO] YOLO Pose runtime: {yolo_pose.runtime} | imgsz={POSE_IMGSZ} | interval={YOLO_INTERVAL}")
    if yolo_hand21 is not None:
        print(f"[INFO] YOLO Hand21 runtime: {yolo_hand21.runtime} | imgsz={POSE_IMGSZ} | interval={YOLO_INTERVAL}")

    # Cachés por detector
    last_pose_kpts, last_pose_boxes = None, None
    last_21_kpts,   last_21_boxes   = None, None

    frame_idx = 0
    consecutive_misses = 0

    while True:
        ok, frame = reader.read(timeout=0.2)  # NO bloqueante; toma último
        if not ok:
            consecutive_misses += 1
            if consecutive_misses % 20 == 0:
                print("[WARN] Sin frame reciente, sigo intentando...")
            if consecutive_misses >= 100:
                print("[INFO] Reabriendo cámara…")
                reader.reopen()
                consecutive_misses = 0
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            continue
        consecutive_misses = 0

        # Espejo para vista “selfie”
        frame = cv2.flip(frame, 1)

        frames += 1
        if frames % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            fps = 10.0 / dt if dt > 0 else fps
            t0 = t1

        display = frame

        # Flags por modo
        do_faces      = mode in (3, 5, 6)
        do_edges      = mode in (2, 6)
        do_hands_pose = mode in (4, 5, 6) and (yolo_pose is not None)
        do_hands_21   = mode in (8,)      and (yolo_hand21 is not None)

        if do_faces:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_casc.detectMultiScale(gray, 1.2, 6, minSize=(60, 60))
                for (x, y, w, h) in faces:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 215, 255), 2)
                    draw_text(display, "Rostro", (x, max(0, y - 8)), 0.6, (0, 215, 255), 2)
            except Exception as e:
                print(f"[WARN] Detección rostro falló: {e}")

        # === Manos (original Pose) ===
        if do_hands_pose:
            try:
                if (frame_idx % YOLO_INTERVAL == 0) or (last_pose_kpts is None):
                    last_pose_kpts, last_pose_boxes = yolo_pose.infer(frame)
                display = yolo_pose.draw(display, last_pose_kpts, last_pose_boxes)
            except Exception as e:
                print(f"[WARN] YOLO Pose falló: {e} (reutilizo última predicción)")

        # === Manos (21 kpts + RPS) SOLO modo 8 ===
        if do_hands_21:
            try:
                if (frame_idx % YOLO_INTERVAL == 0) or (last_21_kpts is None):
                    last_21_kpts, last_21_boxes = yolo_hand21.infer(frame)
                display = yolo_hand21.draw(display, last_21_kpts, last_21_boxes)
            except Exception as e:
                print(f"[WARN] YOLO Hand21 falló: {e} (reutilizo última predicción)")

        if do_edges:
            try:
                _, edges = edges_proc.process(frame, min_area=800)
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                display = cv2.addWeighted(display, 0.85, edges_rgb, 0.45, 0)
            except Exception as e:
                print(f"[WARN] Edges (CUDA) falló: {e} (fallback CPU)")
                try:
                    _, edges = process_image_pipeline(frame, min_area=800)
                    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    display = cv2.addWeighted(display, 0.85, edges_rgb, 0.45, 0)
                except Exception as e2:
                    print(f"[WARN] Edges (CPU) también falló: {e2}")

        # ===== MODO 7: Convoluciones =====
        if mode == 7:
            try:
                conv_name, cfg = CONV_PRESETS[conv_idx]
                display = conv_proc.apply(display, cfg)
            except Exception as e:
                print(f"[WARN] Convolución falló: {e}")

        # HUD (elige runtime según modo)
        hud_conv_name = CONV_PRESETS[conv_idx][0] if mode == 7 else None
        runtime_str = (
            (yolo_pose.runtime if yolo_pose else "N/A") if mode in (4, 5, 6)
            else (yolo_hand21.runtime if yolo_hand21 else "N/A") if mode == 8
            else "N/A"
        )
        draw_hud(
            display,
            fps,
            mode,
            HAS_YOLO,
            runtime_str,
            POSE_IMGSZ,
            YOLO_INTERVAL,
            conv_name=hud_conv_name
        )

        cv2.imshow("CV Demo (Cam)", display)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            out = Path("snapshot.jpg"); cv2.imwrite(str(out), display)
            print(f"[OK] Snapshot guardado en {out.resolve()}")
        elif k in [ord(str(i)) for i in [1, 2, 3, 4, 5, 6, 7, 8]]:
            mode = int(chr(k))
        elif k in (ord(']'),):
            conv_idx = (conv_idx + 1) % len(CONV_PRESETS)
            if mode == 7: print(f"[INFO] Filtro: {CONV_PRESETS[conv_idx][0]}")
        elif k in (ord('['),):
            conv_idx = (conv_idx - 1) % len(CONV_PRESETS)
            if mode == 7: print(f"[INFO] Filtro: {CONV_PRESETS[conv_idx][0]}")
        elif k == ord('n'):
            conv_idx = (conv_idx + 1) % len(CONV_PRESETS)
            if mode == 7: print(f"[INFO] Filtro: {CONV_PRESETS[conv_idx][0]}")
        elif k == ord('p'):
            conv_idx = (conv_idx - 1) % len(CONV_PRESETS)
            if mode == 7: print(f"[INFO] Filtro: {CONV_PRESETS[conv_idx][0]}")

        frame_idx += 1

    reader.stop()
    cv2.destroyAllWindows()

# ============================
# CLI
# ============================

def parse_args():
    ap = argparse.ArgumentParser(description="Demo VC (Modos 1..8; 4-6 manos Pose original espejo corregido; 8 manos 21kpts+RPS mejorado; 7 Convoluciones)")
    ap.add_argument("--image", type=str, help="Ruta a imagen para demo estática")
    ap.add_argument("--camera", type=int, help="Índice de cámara (p.ej. 0)", default=None)
    ap.add_argument("--interval", type=int, help=f"Intervalo de frames para YOLO (def {YOLO_INTERVAL})", default=YOLO_INTERVAL)
    ap.add_argument("--imgsz", type=int, help=f"Tamaño de inferencia (def {POSE_IMGSZ})", default=POSE_IMGSZ)
    ap.add_argument("--trt", action="store_true", help="Usar TensorRT (.engine) para Pose si está disponible")
    return ap.parse_args()


def main():
    print(f"[INFO] Ultralytics YOLO disponible: {HAS_YOLO}")
    print(f"[INFO] CUDA (Torch): {HAS_CUDA} | OpenCV CUDA: {HAS_CV_CUDA}")

    args = parse_args()

    global YOLO_INTERVAL, POSE_IMGSZ, USE_TRT
    YOLO_INTERVAL = max(1, int(args.interval))
    POSE_IMGSZ = max(256, int(args.imgsz))
    USE_TRT = bool(args.trt)

    if args.image:
        run_image_demo(Path(args.image))
    else:
        cam_index = 0 if args.camera is None else args.camera
        run_camera_demo(cam_index)


if __name__ == "__main__":
    main()
