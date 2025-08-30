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
YOLO_INTERVAL = 2     # corre YOLOv8-Pose cada N frames (2 o 3 suele ir bien)
POSE_IMGSZ   = 512    # tamaño de inferencia de pose (512 acelera vs 640)
USE_TRT      = False  # si ya exportaste yolov8n-pose.engine, pon True

# ============================
# CUDA / Torch (GPU)
# ============================
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        # Autotune para tamaños estables
        torch.backends.cudnn.benchmark = True
        # TF32 acelera conv/matmul en Ampere+ sin pérdida visible
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
except Exception:
    HAS_CUDA = False

# ============================
# YOLOv8 Pose (Ultralytics)
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
# Edge processor (GPU si hay OpenCV CUDA)
# ============================
class EdgeProcessor:
    """
    Procesa blur + Canny en GPU si OpenCV CUDA está disponible; si no, usa CPU.
    Mantiene calidad (sin downscale). Devuelve (overlay, edges).
    """

    def __init__(self, canny_th1=80, canny_th2=160, blur_ks=(5, 5), blur_sigma=1.2):
        self.use_gpu = HAS_CV_CUDA
        self.canny_th1 = canny_th1
        self.canny_th2 = canny_th2
        self.blur_ks = blur_ks
        self.blur_sigma = blur_sigma

        if self.use_gpu:
            # Filtros persistentes y GpuMat prealocados
            self._gf = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, self.blur_ks, self.blur_sigma)
            self._canny = cv2.cuda.createCannyEdgeDetector(self.canny_th1, self.canny_th2)
            self._gpu_bgr = cv2.cuda_GpuMat()
            self._gpu_gray = cv2.cuda_GpuMat()
            self._gpu_blur = cv2.cuda_GpuMat()
            self._gpu_edges = cv2.cuda_GpuMat()

    def process(self, bgr, min_area=500):
        if self.use_gpu:
            self._gpu_bgr.upload(bgr)
            # Nota: algunas operaciones CUDA devuelven nuevos GpuMat; mantenemos referencias
            self._gpu_gray = cv2.cuda.cvtColor(self._gpu_bgr, cv2.COLOR_BGR2GRAY)
            self._gpu_blur = self._gf.apply(self._gpu_gray)
            self._gpu_edges = self._canny.detect(self._gpu_blur)
            edges = self._gpu_edges.download()
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, self.blur_ks, self.blur_sigma)
            edges = cv2.Canny(blur, self.canny_th1, self.canny_th2)

        # Contornos en CPU (no hay versión CUDA)
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
# Fallback CPU pipeline (si quisieras llamarla directo)
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
# YOLOv8-Pose -> manos (muñeca) + "esqueleto" local
# ============================
class PoseHandEstimator:
    """
    Usa yolov8n-pose para obtener puntos clave (COCO).
    Dibuja línea codo→muñeca y una caja centrada en muñeca.
    Ejecuta en GPU con FP16 si está disponible. Opción de TensorRT.
    """

    def __init__(self, imgsz=640, conf=0.5, use_trt=False, warmup=True):
        if not HAS_YOLO:
            raise RuntimeError("Ultralytics YOLO no está disponible")
        model_path = "yolov8n-pose.engine" if use_trt else "yolov8n-pose.pt"
        self.model = YOLO(model_path)  # si es .pt se descarga automáticamente
        self.imgsz = imgsz
        self.conf = conf
        self.device = 0 if HAS_CUDA else "cpu"
        self.fp16 = bool(HAS_CUDA)
        self.runtime = "TRT FP16" if use_trt and HAS_CUDA else ("GPU FP16" if HAS_CUDA else "CPU")
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

            for wrist, elbow, side in [
                (left_wrist, left_elbow, "Left"),
                (right_wrist, right_elbow, "Right"),
            ]:
                wx, wy = wrist
                ex, ey = elbow
                if (wx <= 0 and wy <= 0) or (ex <= 0 and ey <= 0):
                    continue

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

                draw_text(canvas, f"Mano {side}", (x0, max(0, y0 - 8)), 0.6, (255, 0, 180), 2)

        return canvas

    def process_and_draw(self, bgr, canvas):
        kpts, boxes = self.infer(bgr)
        return self.draw(canvas, kpts, boxes)

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
        # Evita latencia: buffer mínimo
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap if cap.isOpened() else None

    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        cap = _try_open(backend)
        if cap is not None:
            return cap

    # Último intento sin backend explícito
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

def draw_hud(frame, fps, mode, has_yolo, yolo_runtime, pose_imgsz, yolo_interval):
    y = 24
    draw_text(frame, f"FPS: {fps:.1f}", (10, y)); y += 24
    draw_text(frame, f"Modo {mode}:", (10, y)); y += 24
    texts = {
        1: "1) Base",
        2: "2) Edges (Canny)",
        3: "3) Caras (Haar)",
        4: "4) Manos (YOLOv8-Pose)",
        5: "5) Caras + Manos (YOLOv8-Pose)",
        6: "6) Caras + Manos + Edges (YOLOv8-Pose)",
    }
    draw_text(frame, texts.get(mode, ""), (10, y), 0.6); y += 20
    draw_text(frame, "Teclas: 1..6 | s snapshot | q salir", (10, y), 0.6); y += 20
    if not has_yolo:
        draw_text(frame, "Nota: instala 'ultralytics' para modos 4-6", (10, y), 0.6, (0, 0, 255), 2); y += 20
    y += 2
    draw_text(frame, f"YOLOv8-Pose: {yolo_runtime} | imgsz={pose_imgsz} | interval={yolo_interval}", (10, y), 0.6); y += 20
    draw_text(frame, f"Torch CUDA: {'ON' if HAS_CUDA else 'OFF'}", (10, y), 0.6); y += 20
    draw_text(frame, f"OpenCV CUDA (edges): {'ON' if HAS_CV_CUDA else 'OFF'}", (10, y), 0.6); y += 20


def run_camera_demo(cam_index=0):
    reader = FrameReader(cam_index).start()

    cv2.setUseOptimized(True)
    try:
        # Evitar que OpenCL compita si tienes CUDA
        if HAS_CUDA and hasattr(cv2.ocl, "setUseOpenCL"):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    face_casc = face_detector()
    yolo_hands = PoseHandEstimator(imgsz=POSE_IMGSZ, conf=0.5, use_trt=USE_TRT) if HAS_YOLO else None
    edges_proc = EdgeProcessor()

    # Por defecto: si hay YOLO -> 4; si no -> 3
    mode = 4 if HAS_YOLO else 3

    t0 = time.time(); frames = 0; fps = 0.0
    print("Modos: 1=Base  2=Edges  3=Caras  4=Manos(YOLOv8-Pose)  5=Caras+Manos(YOLOv8-Pose)  6=Caras+Manos+Edges(YOLOv8-Pose)")
    print("Teclas: 1..6 | s snapshot | q salir")
    print(f"[INFO] Torch CUDA: {HAS_CUDA} | OpenCV CUDA: {HAS_CV_CUDA} | Ultralytics: {HAS_YOLO}")
    if yolo_hands is not None:
        print(f"[INFO] YOLOv8-Pose runtime: {yolo_hands.runtime} | imgsz={POSE_IMGSZ} | interval={YOLO_INTERVAL}")

    last_kpts, last_boxes = None, None
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
            # sigue el loop para no congelar
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            continue
        consecutive_misses = 0

        frame = cv2.flip(frame, 1)

        frames += 1
        if frames % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            fps = 10.0 / dt if dt > 0 else fps
            t0 = t1

        # Evita copias innecesarias
        display = frame

        do_faces = mode in (3, 5, 6)
        do_hands_yolo = mode in (4, 5, 6) and (yolo_hands is not None)
        do_edges = mode in (2, 6)

        if do_faces:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_casc.detectMultiScale(gray, 1.2, 6, minSize=(60, 60))
                for (x, y, w, h) in faces:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 215, 255), 2)
                    draw_text(display, "Rostro", (x, max(0, y - 8)), 0.6, (0, 215, 255), 2)
            except Exception as e:
                print(f"[WARN] Detección rostro falló: {e}")

        if do_hands_yolo:
            try:
                # Corre YOLO cada N frames y reutiliza la última predicción
                if (frame_idx % YOLO_INTERVAL == 0) or (last_kpts is None):
                    last_kpts, last_boxes = yolo_hands.infer(frame)
                display = yolo_hands.draw(display, last_kpts, last_boxes)
            except Exception as e:
                print(f"[WARN] YOLO falló: {e} (reutilizo última predicción)")

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

        draw_hud(
            display,
            fps,
            mode,
            HAS_YOLO,
            yolo_hands.runtime if yolo_hands else "N/A",
            POSE_IMGSZ,
            YOLO_INTERVAL,
        )

        cv2.imshow("CV Demo (Cam)", display)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            out = Path("snapshot.jpg"); cv2.imwrite(str(out), display)
            print(f"[OK] Snapshot guardado en {out.resolve()}")
        elif k in [ord(str(i)) for i in [1, 2, 3, 4, 5, 6]]:
            mode = int(chr(k))

        frame_idx += 1

    reader.stop()
    cv2.destroyAllWindows()

# ============================
# CLI
# ============================

def parse_args():
    ap = argparse.ArgumentParser(description="Demo VC (Modos 1..6; manos con YOLOv8-Pose) - Optimizada FPS + Drop Frames")
    ap.add_argument("--image", type=str, help="Ruta a imagen para demo estática")
    ap.add_argument("--camera", type=int, help="Índice de cámara (p.ej. 0)", default=None)
    ap.add_argument("--interval", type=int, help=f"Intervalo de frames para YOLO (def {YOLO_INTERVAL})", default=YOLO_INTERVAL)
    ap.add_argument("--imgsz", type=int, help=f"Tamaño de inferencia pose (def {POSE_IMGSZ})", default=POSE_IMGSZ)
    ap.add_argument("--trt", action="store_true", help="Usar TensorRT (.engine) si está disponible")
    return ap.parse_args()


def main():
    print(f"[INFO] Ultralytics YOLO disponible: {HAS_YOLO}")
    print(f"[INFO] CUDA (Torch): {HAS_CUDA} | OpenCV CUDA: {HAS_CV_CUDA}")

    args = parse_args()

    # Permitir override por CLI
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
