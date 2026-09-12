"""
A real screen, with rails.

The optional one: everything else in the repo runs on numpy alone, and this
needs a browser. It exists because "attach it to anything" is easy to claim and
a live page is where the claim gets tested.

The rails are not decoration. A network that drifts along contrast will
eventually land on something, and a click is an action in the world:

  no keyboard          it cannot type, so it cannot fill a field, answer a
                       prompt, or confirm anything
  no wallet, no keys   the browser gets a blank profile: no extension, no
                       stored credential, no logged-in session
  an allowlist         not a keyword blocklist. Keyword filters do not hold.
  every click vetted   anything reading as submit, upload, sign-in, download,
                       purchase or delete is refused before it lands
  an anti-stall budget if the cursor has not moved in N steps the view scrolls,
                       so a dead end does not become a permanent home

Install the extra first:  pip install "mousecortex[screen]" && playwright install chromium
"""
from __future__ import annotations

import io
import re
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from ..readout import Action
from ..task import Task

VETO = re.compile(
    r"(submit|upload|sign[ -]?in|log[ -]?in|sign[ -]?up|register|download|"
    r"donate|delete|remove|buy|checkout|pay|subscribe|confirm|connect wallet)",
    re.I)

PROBE_JS = """(p) => {
  const e = document.elementFromPoint(p.x, p.y);
  if (!e) return {tag: null};
  const a = e.closest('a');
  const t = (e.innerText || '').slice(0, 80);
  return {tag: e.tagName, href: a ? a.href : null,
          text: a ? (a.innerText || '').slice(0, 120) : t};
}"""


class ScreenTask(Task):
    """Point at a live page. Reward is yours to define."""

    name = "screen"

    def __init__(self,
                 url: str,
                 allow: Sequence[str],
                 reward_fn: Optional[Callable[["ScreenTask", Action], float]] = None,
                 size: Tuple[int, int] = (1280, 720),
                 gaze: Tuple[int, int] = (300, 210),
                 headless: bool = True,
                 allow_clicks: bool = True,
                 stall_steps: int = 15,
                 max_steps: int = 200):
        if not allow:
            raise ValueError("an allowlist is required - an empty one means the "
                             "whole internet, which is not a rail")
        self.url = url
        self.allow = tuple(allow)
        self.reward_fn = reward_fn
        self.w, self.h = size
        self.gaze = gaze
        self.headless = headless
        self.allow_clicks = allow_clicks
        self.stall_steps = int(stall_steps)
        self.max_steps = int(max_steps)

        self.cursor = [self.w / 2.0, self.h / 2.0]
        self.clicks = self.hops = self.vetoed = self.stalls = 0
        self._recent: list = []
        self._steps = 0
        self._frame = np.zeros((self.h, self.w), dtype=np.float32)
        self._page = self._browser = self._pw = None

    # ------------------------------------------------------------ lifecycle --
    def open(self) -> "ScreenTask":
        from playwright.sync_api import sync_playwright
        from PIL import Image  # noqa: F401  (checked here, used in observe)

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        ctx = self._browser.new_context(viewport={"width": self.w, "height": self.h})
        self._page = ctx.new_page()
        self._page.goto(self.url, wait_until="domcontentloaded", timeout=45000)
        self._page.wait_for_timeout(1200)
        return self

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._pw is not None:
            self._pw.stop()
        self._page = self._browser = self._pw = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ----------------------------------------------------------------- task --
    def reset(self) -> None:
        if self._page is None:
            self.open()
        self.cursor = [self.w / 2.0, self.h / 2.0]
        self._page.mouse.move(*self.cursor)
        self._recent.clear()
        self._steps = 0

    def observe(self) -> np.ndarray:
        from PIL import Image
        raw = self._page.screenshot(type="jpeg", quality=60)
        img = Image.open(io.BytesIO(raw)).convert("L")
        self._frame = np.asarray(img, dtype=np.float32) / 255.0
        return self._frame

    def focus(self):
        return (self.cursor[0], self.cursor[1])

    def window(self):
        return self.gaze

    def act(self, action: Action) -> float:
        self._steps += 1
        page = self._page

        y = self.cursor[1] + action.dy
        if y > self.h - 20:                        # drive past the bottom scrolls
            page.mouse.wheel(0, 260)
            page.wait_for_timeout(200)
            y = self.h - 80
        elif y < 20:
            page.mouse.wheel(0, -260)
            page.wait_for_timeout(200)
            y = 80

        self.cursor[0] = float(np.clip(self.cursor[0] + action.dx, 4, self.w - 4))
        self.cursor[1] = float(np.clip(y, 4, self.h - 4))
        page.mouse.move(*self.cursor)

        self._recent.append(tuple(self.cursor))
        self._recent = self._recent[-self.stall_steps:]
        if len(self._recent) == self.stall_steps:
            sp = np.array(self._recent)
            if np.abs(sp - sp[0]).max() < 30:
                page.mouse.wheel(0, 420)
                page.wait_for_timeout(300)
                self.stalls += 1
                self._recent.clear()

        self.last_click = None
        if action.commit and self.allow_clicks:
            self._try_click()

        return float(self.reward_fn(self, action)) if self.reward_fn else 0.0

    def _try_click(self) -> None:
        info = self._page.evaluate(PROBE_JS, {"x": self.cursor[0], "y": self.cursor[1]})
        href = info.get("href") or ""
        text = info.get("text") or ""
        if VETO.search(text) or VETO.search(href):
            self.vetoed += 1
            self.last_click = ("vetoed", text[:60])
            return
        if href and not any(d in href for d in self.allow):
            self.vetoed += 1
            self.last_click = ("off-allowlist", href[:80])
            return
        self._page.mouse.click(*self.cursor)
        self.clicks += 1
        self.last_click = ("clicked", text[:60])
        if href:
            self._page.wait_for_timeout(1500)
            self.hops += 1
            self.cursor = [self.w / 2.0, self.h / 2.0]
            self._page.mouse.move(*self.cursor)

    def done(self) -> bool:
        return self._steps >= self.max_steps

    def report(self) -> dict:
        return {"steps": self._steps, "clicks": self.clicks, "hops": self.hops,
                "vetoed": self.vetoed, "stall_scrolls": self.stalls}
