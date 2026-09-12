"""
The eye: anything two-dimensional, turned into drive.

The cortex in this repo is a visual area, so the one thing it accepts is a
field of numbers laid out in space. That is less of a restriction than it
sounds. A screenshot is a field. So is a game board, an order book rendered as
a heatmap, a spectrogram, an attention matrix, the occupancy grid of a robot,
or a 2D projection of an embedding. If you can draw it, this can look at it.

Pixels enter at the measured receptive-field centres: the digital twin reports,
for every coregistered cell, which point of stimulus space it looks at. A fly
connectome has to lay its hex columns onto the screen by hand; here the
retinotopy is data.

Two components of drive, because a V1 cell answers to two things:

  contrast   local contrast at the receptive-field centre, which tells the
             population where it is. Absolute brightness is kept only as a weak
             presence signal: a cell inside a uniformly bright box has nothing
             to say, and driving on flat luminance left the edge contribution
             at 0.7% of total drive.

  edge       these cells are tuned to a direction of motion, so a still frame
             drives every one of them the same way and the readout just reports
             the sampling bias of the dataset. A cell preferring direction th
             fires on a luminance edge whose bright side lies opposite th, the
             leading edge it would see if the stimulus were moving that way. So
             the drive is the image gradient projected onto each cell's own
             preferred direction, half-wave rectified.

The gradient is measured. The projection is the standard model of a
direction-selective response. Neither is fitted.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import params
from .graph import Graph


class Retina:
    """Retinotopic sampling of a 2D field at measured receptive-field centres."""

    def __init__(self, graph: Graph, cc_min: float = params.CC_MIN):
        rf = graph.rf
        usable = np.isfinite(rf[:, 0]) & np.isfinite(rf[:, 1])
        if np.isfinite(graph.cc_abs).any():
            usable &= np.nan_to_num(graph.cc_abs, nan=0.0) >= cc_min
        self.idx = np.flatnonzero(usable).astype(np.int64)
        if self.idx.size == 0:
            raise ValueError(
                "no cell in this graph has both a receptive field and a twin "
                "score above cc_min - lower cc_min, or rebuild with tuning data")

        rx, ry = rf[self.idx, 0], rf[self.idx, 1]
        # normalise into the patch the population actually covers, so it spans
        # the whole view instead of a corner of it
        self.u = ((rx - rx.min()) / (rx.max() - rx.min() + 1e-9)).astype(np.float32)
        self.v = ((ry - ry.min()) / (ry.max() - ry.min() + 1e-9)).astype(np.float32)

        self.pref_dir = np.deg2rad(np.nan_to_num(graph.pref_dir[self.idx])).astype(np.float32)
        self.gDSI = np.nan_to_num(graph.gDSI[self.idx]).astype(np.float32)
        self.cc_abs = np.nan_to_num(graph.cc_abs[self.idx]).astype(np.float32)

        # a cell fires on an edge whose bright side lies opposite its preferred
        # direction; screen y grows downward while stimulus angle grows the
        # other way, hence the minus
        self._ux = np.cos(self.pref_dir).astype(np.float32)
        self._uy = (-np.sin(self.pref_dir)).astype(np.float32)

    def __len__(self) -> int:
        return int(self.idx.size)

    def describe(self) -> str:
        tuned = int((self.gDSI >= params.GDSI_MIN).sum())
        return f"retina: {len(self):,} cells, {tuned:,} direction-tuned"

    # ------------------------------------------------------------------ look --
    def look(self, field: np.ndarray,
             focus: Optional[Tuple[float, float]] = None,
             window: Optional[Tuple[float, float]] = None,
             max_hz: float = params.CONTRAST_HZ,
             edge_hz: float = params.EDGE_HZ,
             lum_hz: float = params.LUMINANCE_HZ,
             spont_hz: float = params.SPONT_HZ,
             grad_px: int = params.GRAD_PX,
             gain: float = 1.0) -> dict:
        """
        Sample a field and return drive rates, ready for Cortex.run.

        field   2D array, any scale; values outside 0..1 are normalised
        focus   (x, y) centre of gaze in field pixels. None looks at all of it.
        window  (w, h) of the gaze in field pixels. None looks at all of it.
        gain    overall excitability, e.g. from a neuromodulator
        """
        img = np.asarray(field, dtype=np.float32)
        if img.ndim == 3:                       # colour in, luminance out
            img = img[..., :3].mean(axis=2)
        if img.ndim != 2:
            raise ValueError(f"a field has to be 2D, got shape {img.shape}")
        hi = float(img.max())
        if hi > 1.0:
            img = img / (hi + 1e-9)

        H, W = img.shape
        fw, fh = (float(W), float(H)) if window is None else (float(window[0]), float(window[1]))
        cx, cy = (W / 2.0, H / 2.0) if focus is None else (float(focus[0]), float(focus[1]))

        def at(u, v):
            px = np.clip((cx - fw / 2 + u * fw).astype(np.int32), 0, W - 1)
            py = np.clip((cy - fh / 2 + v * fh).astype(np.int32), 0, H - 1)
            return img[py, px]

        du = grad_px / max(fw, 1.0)
        dv = grad_px / max(fh, 1.0)
        lum = at(self.u, self.v)
        gx = at(np.clip(self.u + du, 0, 1), self.v) - at(np.clip(self.u - du, 0, 1), self.v)
        gy = at(self.u, np.clip(self.v + dv, 0, 1)) - at(self.u, np.clip(self.v - dv, 0, 1))

        edge = np.clip(-(gx * self._ux + gy * self._uy), 0, None) * self.gDSI * 4.0
        contrast = np.hypot(gx, gy)

        rate = (spont_hz
                + np.clip(lum, 0, 1) * lum_hz
                + np.clip(contrast, 0, 1) * max_hz
                + np.clip(edge, 0, 1) * edge_hz) * float(gain)
        return {tuple(self.idx): rate.astype(np.float32)}


def render_points(points, size=(160, 120), radius: float = 6.0,
                  weights=None) -> np.ndarray:
    """
    Draw a list of (x, y) points as a field, for tasks that are not images.

    A convenience, not part of the model: it exists so that anything you can
    express as "these things are at these places, and these matter more" can be
    handed to a visual cortex without writing a renderer.
    """
    w, h = int(size[0]), int(size[1])
    out = np.zeros((h, w), dtype=np.float32)
    pts = np.atleast_2d(np.asarray(points, dtype=np.float32))
    if pts.size == 0:
        return out
    wts = np.ones(len(pts), np.float32) if weights is None else np.asarray(weights, np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for (px, py), k in zip(pts, wts):
        d2 = (xx - px) ** 2 + (yy - py) ** 2
        out += k * np.exp(-d2 / (2.0 * radius ** 2))
    m = out.max()
    return out / m if m > 0 else out
