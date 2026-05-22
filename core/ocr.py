"""
License-plate OCR — fast, accurate, tilt-tolerant, low-light resilient.

Strategy (progressive, stops as soon as confidence is high enough):
  1. Upscale crop to ≥140 px tall
  2. FAST PATH  — run EasyOCR recognize() directly on upscaled image
                  (skips slow CRAFT text-detection step, ~2-3× faster)
  3. SLOW PATH  — if confidence still < 0.72, try two preprocessed variants:
                  a) CLAHE + unsharp-mask          (general purpose)
                  b) deskew + adaptive threshold   (tilted / dusty / dark)
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np

from . import config

log = logging.getLogger(__name__)

_PLATE_OK = re.compile(r"[A-Z0-9]")
_ALLOWED  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# Digit → Letter  (applied at letter positions: state code, series)
_D2L = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "7": "Z", "8": "B"})
# Letter → Digit  (applied at digit positions: district number, serial)
_L2D = str.maketrans({"B": "8", "D": "0", "G": "6", "I": "1", "J": "1",
                       "L": "1", "O": "0", "Q": "0", "S": "5", "Z": "2"})

_FAST_CONF_THRESHOLD = 0.72   # accept fast-path result above this, skip slow path
_GOOD_CONF_THRESHOLD = 0.82   # stop trying further variants above this


class PlateOCR:
    def __init__(self, gpu: bool = False, languages: tuple[str, ...] = ("en",)):
        import easyocr
        self.reader    = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
        self.min_conf  = float(config.get("thresholds", "ocr_conf",        default=0.55))
        self.min_chars = int  (config.get("thresholds", "min_plate_chars",  default=5))
        log.info("EasyOCR ready (gpu=%s)", gpu)

    # ------------------------------------------------------------------ public

    def read(self, plate_img: np.ndarray) -> tuple[Optional[str], float]:
        if plate_img is None or plate_img.size == 0:
            return None, 0.0

        # Step 1 — upscale once; all variants share this base
        base = self._upscale(plate_img)

        # Step 2 — FAST PATH: recognize() skips CRAFT detection, ~2-3× faster
        best_plate, best_conf = self._ocr_recognize(base)
        if best_conf >= _FAST_CONF_THRESHOLD:
            return best_plate, best_conf

        # Step 3 — SLOW PATH: try preprocessed variants for difficult plates
        for variant in self._preprocess_variants(base):
            plate, conf = self._ocr_recognize(variant)
            if plate is not None and conf > best_conf:
                best_plate, best_conf = plate, conf
            if best_conf >= _GOOD_CONF_THRESHOLD:
                break

        return best_plate, best_conf

    # ------------------------------------------------------------------ OCR call

    def _ocr_recognize(self, img: np.ndarray) -> tuple[Optional[str], float]:
        """
        Call EasyOCR recognize() — bypasses CRAFT text-detection entirely.
        Treats the whole image as one text region (correct for a plate crop).
        """
        h, w = img.shape[:2]
        try:
            # recognize() takes pre-defined text regions, skipping CRAFT
            results = self.reader.recognize(
                img,
                horizontal_list=[[0, w, 0, h]],
                free_list=[],
                allowlist=_ALLOWED,
                detail=1,
                paragraph=False,
            )
        except Exception:
            # Fallback: full readtext if recognize() is unavailable
            try:
                results = self.reader.readtext(img, detail=1, paragraph=False,
                                               allowlist=_ALLOWED)
            except Exception as exc:
                log.debug("OCR error: %s", exc)
                return None, 0.0

        if not results:
            return None, 0.0

        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

        pieces, confs = [], []
        for _box, text, conf in results:
            if conf < self.min_conf * 0.5:
                continue
            cleaned = "".join(ch for ch in text.upper() if _PLATE_OK.match(ch))
            if cleaned:
                pieces.append(cleaned)
                confs.append(float(conf))

        if not pieces:
            return None, 0.0

        plate    = self._postfix("".join(pieces))
        avg_conf = float(np.mean(confs))

        if len(plate) < self.min_chars or avg_conf < self.min_conf:
            return None, avg_conf

        return plate, avg_conf

    # ------------------------------------------------------------------ preprocessing

    @staticmethod
    def _upscale(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h < 140:
            scale = 140 / max(h, 1)
            img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)
        return img

    @staticmethod
    def _preprocess_variants(img: np.ndarray) -> list[np.ndarray]:
        """Two enhanced grayscale variants for difficult plates."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

        # Brightness correction
        mean_val = float(np.mean(gray))
        if mean_val < 80:
            gamma = 0.40
        elif mean_val < 110:
            gamma = 0.65
        elif mean_val > 210:
            gamma = 1.8
        else:
            gamma = None
        if gamma is not None:
            lut  = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
            gray = cv2.LUT(gray, lut)

        # Remove dust / salt-and-pepper noise
        gray = cv2.medianBlur(gray, 3)

        # Variant A — CLAHE + fast unsharp-mask  (good for most plates)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6))
        va    = clahe.apply(gray)
        blur  = cv2.GaussianBlur(va, (5, 5), 0)
        va    = cv2.addWeighted(va, 1.6, blur, -0.6, 0)

        # Variant B — deskew + adaptive threshold  (tilted / uneven lighting)
        vb    = PlateOCR._deskew(gray)
        vb    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(vb)
        vb    = cv2.adaptiveThreshold(vb, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, blockSize=15, C=7)

        return [va, vb]

    # ------------------------------------------------------------------ deskew

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return PlateOCR._deskew_hough(gray)
        h, w  = gray.shape[:2]
        best  = None
        best_score = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < h * w * 0.02:
                continue
            rx, ry, rw, rh = cv2.boundingRect(c)
            aspect = rw / max(rh, 1)
            if aspect < 1.5:
                continue
            s = area * aspect
            if s > best_score:
                best_score = s
                best = c
        if best is None:
            return PlateOCR._deskew_hough(gray)
        angle = cv2.minAreaRect(best)[2]
        if angle < -45:
            angle += 90
        if abs(angle) < 1.5:
            return gray
        angle = max(-45.0, min(45.0, angle))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _deskew_hough(gray: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=40)
        if lines is None:
            return gray
        angles = []
        for line in lines[:20]:
            a = np.degrees(line[0][1]) - 90
            if -45 < a < 45:
                angles.append(a)
        if not angles or abs(float(np.median(angles))) < 1.5:
            return gray
        angle = float(np.median(angles))
        h, w  = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # ------------------------------------------------------------------ postfix

    @staticmethod
    def _postfix(plate: str) -> str:
        """Position-aware letter/digit fix for Indian format SS DD (L|LL) NNNN."""
        n = len(plate)
        # Indian plate formats:
        #   8  → KL07B123   SS DD L NNN
        #   9  → KL07BX123  SS DD LL NNN  or  SS DD L NNNN
        #   10 → KL07BX1234 SS DD LL NNNN
        if n == 8:
            lp = {0, 1, 4};      dp = {2, 3, 5, 6, 7}
        elif n == 9:
            lp = {0, 1, 4};      dp = {2, 3, 5, 6, 7, 8}
        elif n == 10:
            lp = {0, 1, 4, 5};   dp = {2, 3, 6, 7, 8, 9}
        elif n >= 6:
            lp = {0, 1};         dp = {2, 3} | set(range(max(4, n - 4), n))
        else:
            return plate
        result = list(plate)
        for i, ch in enumerate(plate):
            if i in lp and ch.isdigit():   result[i] = ch.translate(_D2L)
            elif i in dp and ch.isalpha(): result[i] = ch.translate(_L2D)
        return "".join(result)
