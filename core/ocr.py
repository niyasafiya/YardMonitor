"""
License-plate OCR — PaddleOCR (PP-OCRv4) backend.

Strategy:
  1. Upscale to ≥200 px tall AND ≥900 px wide
  2. Add white border so edge characters are never clipped
  3. Run PaddleOCR on base image + 4 preprocessed variants (5 passes total)
  4. Confidence-weighted majority vote at character level
  5. Position-aware letter/digit fix for Indian plates (SS DD L NNNN)

PaddleOCR vs EasyOCR:
  - ~4-6x faster per crop (PP-OCRv4 mobile vs CRAFT+CRNN)
  - Better accuracy on digit/letter confusion (1/4, 3/9, O/0, R/B)
  - No character-split artefacts (R→Y+4 was an EasyOCR segmentation bug)
  - Models auto-download to ~/.paddleocr/ on first run (~50 MB)
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config

_DEBUG_CROPS = bool(int(os.getenv("YM_DEBUG_CROPS", "0")))
_DEBUG_DIR   = Path("data/debug_crops")

log = logging.getLogger(__name__)

_PLATE_OK = re.compile(r"[A-Z0-9]")

# Digit → Letter  (applied at letter positions: state code, series)
_D2L = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "7": "Z", "8": "B"})
# Letter → Digit  (applied at digit positions: district number, serial)
_L2D = str.maketrans({"B": "8", "D": "0", "G": "6", "I": "1", "J": "1",
                       "L": "1", "O": "0", "Q": "0", "S": "5", "Z": "2"})

# PaddleOCR confidence scores are calibrated differently from EasyOCR —
# 0.25 keeps borderline candidates for the majority vote while filtering noise.
_VARIANT_MIN_CONF = 0.25


class PlateOCR:
    def __init__(self, gpu: bool = False, languages: tuple[str, ...] = ("en",)):
        from paddleocr import PaddleOCR
        # use_angle_cls=False: plates are always horizontal after dewarp;
        # skipping it saves one model inference per pass.
        self._paddle = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=gpu,
            show_log=False,
        )
        self.min_conf  = float(config.get("thresholds", "ocr_conf",       default=0.40))
        self.min_chars = int  (config.get("thresholds", "min_plate_chars", default=4))
        log.info("PaddleOCR ready (gpu=%s)", gpu)

    # ------------------------------------------------------------------ public

    def read(self, plate_img: np.ndarray) -> tuple[Optional[str], float]:
        if plate_img is None or plate_img.size == 0:
            return None, 0.0
        h, w = plate_img.shape[:2]
        # Plates are always wider than tall (≥2:1). Reject portrait crops.
        if w < 2 * h:
            return None, 0.0

        base = self._upscale(plate_img)
        base = self._add_border(base, px=15)

        # Pass 1-4: preprocessed variants with det=False (whole crop = one text region)
        # Pass 5: base image with det=True (PaddleOCR detection finds character bboxes)
        candidates: list[tuple[str, float]] = []

        for variant in [base] + self._preprocess_variants(base):
            plate, conf = self._ocr_once(variant, det=False)
            if plate is not None:
                candidates.append((plate, conf))

        # det=True pass on the colour base — EAST/DB detector sometimes finds
        # characters the full-image pass misses on busy backgrounds.
        plate_det, conf_det = self._ocr_once(base, det=True)
        if plate_det is not None:
            candidates.append((plate_det, conf_det))

        if not candidates:
            return None, 0.0

        plate, conf = self._majority_vote(candidates)
        if plate is None:
            return None, 0.0

        # Final postfix run on the voted result — catches split patterns that
        # survived character-level voting (e.g. "Y" wins pos-4 and "4" wins pos-5).
        fixed = self._postfix(plate)
        if fixed != plate:
            log.info("Post-vote correction: %s → %s", plate, fixed)
            plate = fixed

        if len(plate) < self.min_chars:
            if _DEBUG_CROPS:
                _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(_DEBUG_DIR / f"REJECT_{int(time.time()*1000)}.jpg"), base)
            return None, 0.0

        if _DEBUG_CROPS:
            _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(_DEBUG_DIR / f"{plate}_{int(time.time()*1000)}.jpg"), base)
            cv2.imwrite(str(_DEBUG_DIR / f"{plate}_{int(time.time()*1000)}_orig.jpg"), plate_img)

        return plate, conf

    # ------------------------------------------------------------------ PaddleOCR call

    def _ocr_once(self, img: np.ndarray, det: bool = False) -> tuple[Optional[str], float]:
        """Run one PaddleOCR pass. Returns (plate_string, avg_confidence) or (None, 0)."""
        # PaddleOCR expects a 3-channel BGR image.
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        try:
            result = self._paddle.ocr(img, det=det, rec=True, cls=False)
        except Exception as exc:
            log.debug("PaddleOCR error: %s", exc)
            return None, 0.0

        if not result or result[0] is None:
            return None, 0.0

        rows = result[0]
        if not rows:
            return None, 0.0

        text_items = self._parse_rows(rows, det=det)
        if not text_items:
            return None, 0.0

        pieces, confs = [], []
        for text, conf in text_items:
            if conf < _VARIANT_MIN_CONF:
                continue
            cleaned = "".join(ch for ch in text.upper() if _PLATE_OK.match(ch))
            if cleaned:
                pieces.append(cleaned)
                confs.append(conf)

        if not pieces:
            return None, 0.0

        plate    = self._postfix("".join(pieces))
        avg_conf = float(np.mean(confs))

        if len(plate) < self.min_chars:
            return None, avg_conf
        if avg_conf < max(self.min_conf * 0.7, _VARIANT_MIN_CONF):
            return None, avg_conf

        return plate, avg_conf

    @staticmethod
    def _parse_rows(rows: list, det: bool) -> list[tuple[str, float]]:
        """Convert raw PaddleOCR rows to sorted (text, confidence) list.

        PaddleOCR result format differs by mode:
          det=True  — each row: [[box_pts], ('text', conf)]
          det=False — each row: ['text', conf]  OR  ('text', conf)
        """
        parsed: list[tuple[float, str, float]] = []  # (x_center, text, conf)
        for row in rows:
            try:
                if det:
                    box, rec = row[0], row[1]
                    text = rec[0]
                    conf = float(rec[1])
                    x_ctr = (float(box[0][0]) + float(box[1][0])) / 2
                else:
                    # row may be [text, conf] or (text, conf)
                    text = row[0]
                    conf = float(row[1])
                    x_ctr = 0.0
                parsed.append((x_ctr, str(text), conf))
            except (IndexError, TypeError, ValueError):
                continue
        parsed.sort(key=lambda t: t[0])
        return [(t, c) for _, t, c in parsed]

    # ------------------------------------------------------------------ voting

    @staticmethod
    def _majority_vote(candidates: list[tuple[str, float]]) -> tuple[Optional[str], float]:
        """Confidence-weighted character-level majority vote across all passes.

        1. Pick the plate length whose candidates have the best count × mean-confidence.
           A character-split (R→Y+4) inflates length by 1 and lowers confidence, so
           the correct shorter reading wins even if it has fewer raw votes.
        2. At each character position, sum confidence scores per character choice —
           the character with the highest total wins.
        """
        if not candidates:
            return None, 0.0
        if len(candidates) == 1:
            return candidates[0]

        len_stats: dict[int, list[float]] = {}
        for p, c in candidates:
            len_stats.setdefault(len(p), []).append(c)

        target_len = max(
            len_stats,
            key=lambda n: len(len_stats[n]) * float(np.mean(len_stats[n]))
        )
        matching = [(p, c) for p, c in candidates if len(p) == target_len]

        if not matching:
            return max(candidates, key=lambda x: x[1])

        voted = []
        for pos in range(target_len):
            char_scores: dict[str, float] = {}
            for p, conf in matching:
                ch = p[pos]
                char_scores[ch] = char_scores.get(ch, 0.0) + conf
            voted.append(max(char_scores, key=char_scores.get))

        plate    = "".join(voted)
        avg_conf = float(np.mean([c for _, c in matching]))
        return plate, avg_conf

    # ------------------------------------------------------------------ preprocessing

    @staticmethod
    def _upscale(img: np.ndarray) -> np.ndarray:
        """Scale up to ≥200 px tall AND ≥900 px wide using LANCZOS4 for sharpness."""
        h, w = img.shape[:2]
        scale = max(200 / max(h, 1), 900 / max(w, 1), 1.0)
        if scale > 1.0:
            new_w = min(int(w * scale), 2000)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return img

    @staticmethod
    def _add_border(img: np.ndarray, px: int = 15) -> np.ndarray:
        val = (255, 255, 255) if img.ndim == 3 else 255
        return cv2.copyMakeBorder(img, px, px, px, px,
                                  cv2.BORDER_CONSTANT, value=val)

    @staticmethod
    def _preprocess_variants(img: np.ndarray) -> list[np.ndarray]:
        """
        Four preprocessed variants.  PaddleOCR is more robust than EasyOCR so
        fewer variants are needed — and the overhead is low enough to run all 4.

        A — CLAHE + unsharp mask           general purpose / low contrast
        B — bilateral + Otsu              sharp edges, clean binary
        C — inverted Otsu                  white-on-dark / reflective plates
        D — morph close + adaptive thr    fills ink breaks from dirt/damage
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

        # Shared brightness correction via gamma LUT
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
        gray = cv2.medianBlur(gray, 3)

        # A — CLAHE + unsharp mask
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6))
        va    = clahe.apply(gray)
        blur  = cv2.GaussianBlur(va, (5, 5), 0)
        va    = cv2.addWeighted(va, 1.6, blur, -0.6, 0)

        # B — bilateral filter + Otsu binary
        vb = cv2.bilateralFilter(gray, 11, 17, 17)
        _, vb = cv2.threshold(vb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # C — inverted Otsu  (white-on-dark / metallic reflective plates)
        _, vc = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # D — morph close + adaptive threshold  (fills breaks from dirt or wear)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        vd = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        vd = cv2.adaptiveThreshold(vd, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, blockSize=11, C=5)

        # Return as BGR so PaddleOCR doesn't have to convert internally
        return [cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) for v in [va, vb, vc, vd]]

    # ------------------------------------------------------------------ postfix

    _PLATE_CORE = re.compile(r'[A-Z]{2}\d{2}[A-Z]{1,3}\d{3,4}')

    # OCR character-split recovery: some OCR engines segment ONE letter into
    # TWO tokens when strokes are misdetected.
    #   R → Y + 4  (R's loop ≈ Y, diagonal leg ≈ 4)
    #   P → F + 7  (P's closed loop ≈ F, open bottom ≈ 7)
    _CHAR_SPLIT: dict = {("Y", "4"): "R", ("F", "7"): "P"}

    @staticmethod
    def _postfix(plate: str) -> str:
        """Position-aware letter/digit fix for Indian format SS DD (L|LL) NNNN."""
        # Extract plate pattern from longer strings (bumper sticker noise etc.)
        if len(plate) > 10:
            m = PlateOCR._PLATE_CORE.search(plate)
            if m:
                plate = m.group(0)

        # Character-split collapse
        if (len(plate) == 9
                and plate[:2].isalpha()
                and plate[2:4].isdigit()
                and plate[4].isalpha()):
            pair = (plate[4], plate[5])
            if pair in PlateOCR._CHAR_SPLIT:
                plate = plate[:4] + PlateOCR._CHAR_SPLIT[pair] + plate[6:]
                log.debug("Char-split recovery: %s+%s → %s",
                          pair[0], pair[1], PlateOCR._CHAR_SPLIT[pair])

        n = len(plate)
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
