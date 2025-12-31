"""
Video processing utilities.

Currently includes a small script to overlay all frames of a video into a
single image (useful for producing motion trails). Edit `main()` to change
parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _cv2_import_error = exc


OverlayMode = Literal["max", "mean", "add", "ema"]


def overlay_video_frames(
    video_path: str | Path,
    output_path: str | Path,
    *,
    mode: OverlayMode = "max",
    stagger: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    scale: float = 1.0,
    ema_alpha: float = 0.08,
) -> Path:
    """
    Overlay frames of a video into a single image and write it to disk.

    Args:
        video_path: Path to the input video.
        output_path: Path to write the final overlay image (e.g. .png).
        mode: How to combine frames:
            - "max": pixelwise max across frames (good for bright trails)
            - "mean": average of frames
            - "add": sum frames then normalize to [0, 1]
            - "ema": exponential moving average with `ema_alpha`
        stagger: Overlay frames at indices `start_frame + k * stagger` (>= 1).
        start_frame: Start from this frame index (>= 0).
        end_frame: Stop before this frame index (exclusive). If None, read to end.
        scale: Resize each frame by this factor (e.g. 0.5). Must be > 0.
        ema_alpha: EMA update weight (0 < alpha <= 1) when mode="ema".

    Returns:
        The resolved output path.
    """
    if cv2 is None:  # pragma: no cover
        raise ImportError(
            "OpenCV (cv2) is required for this script. Install opencv-python."
        ) from _cv2_import_error

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if stagger < 1:
        raise ValueError("stagger must be >= 1")
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    if end_frame is not None and end_frame < start_frame:
        raise ValueError("end_frame must be >= start_frame")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    if mode == "ema" and not (0.0 < ema_alpha <= 1.0):
        raise ValueError("ema_alpha must be in (0, 1]")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_index = 0
    processed = 0

    sum_accum: Optional[np.ndarray] = None
    max_accum: Optional[np.ndarray] = None

    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame

    while True:
        if end_frame is not None and frame_index >= end_frame:
            break

        ok, frame_bgr = cap.read()
        if not ok:
            break

        if (frame_index - start_frame) % stagger != 0:
            frame_index += 1
            continue

        if scale != 1.0:
            h, w = frame_bgr.shape[:2]
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame_f = frame_bgr.astype(np.float32) / 255.0

        if mode == "max":
            if max_accum is None:
                max_accum = frame_f
            else:
                np.maximum(max_accum, frame_f, out=max_accum)
        elif mode == "mean":
            if sum_accum is None:
                sum_accum = frame_f
            else:
                sum_accum += frame_f
        elif mode == "add":
            if sum_accum is None:
                sum_accum = frame_f
            else:
                sum_accum += frame_f
        elif mode == "ema":
            if sum_accum is None:
                sum_accum = frame_f
            else:
                # EMA: acc = (1-a)*acc + a*frame
                sum_accum *= 1.0 - ema_alpha
                sum_accum += ema_alpha * frame_f
        else:
            raise ValueError(f"Unknown mode: {mode}")

        processed += 1
        frame_index += 1

    cap.release()

    if processed == 0:
        raise RuntimeError("No frames processed (check start/end/sample_every).")

    if mode == "max":
        assert max_accum is not None
        out_f = max_accum
    elif mode == "mean":
        assert sum_accum is not None
        out_f = sum_accum / float(processed)
    elif mode == "add":
        assert sum_accum is not None
        max_val = float(sum_accum.max())
        out_f = sum_accum if max_val <= 0 else (sum_accum / max_val)
    elif mode == "ema":
        assert sum_accum is not None
        out_f = sum_accum
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out_bgr = (np.clip(out_f, 0.0, 1.0) * 255.0).astype(np.uint8)
    ok = cv2.imwrite(str(output_path), out_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {output_path}")

    return output_path.resolve()


def main() -> None:
    video = Path("plots/examples/four_drones_head_on/four_drones_head_on.mov")
    out = Path("plots/examples/four_drones_head_on/four_drones_head_on_overlay.png")

    mode: OverlayMode = "max"
    stagger = 10 
    start_frame = 50 
    end_frame: Optional[int] = None
    scale = 1.0
    ema_alpha = 0.08

    out = overlay_video_frames(
        video,
        out,
        mode=mode,
        stagger=stagger,
        start_frame=start_frame,
        end_frame=end_frame,
        scale=scale,
        ema_alpha=ema_alpha,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
