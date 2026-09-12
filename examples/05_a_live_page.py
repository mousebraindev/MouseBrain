"""
The optional one: point the brain at a real page.

Needs the browser extra:

    pip install "mousecortex[screen]"
    python -m playwright install chromium
    python examples/05_a_live_page.py

The rails live in cortex/tasks/screen.py and they are not decoration: no
keyboard, no wallet, no stored session, an allowlist rather than a keyword
blocklist, and every click checked before it lands.

There is no reward here. Untrained and unpaid, the cursor drifts along contrast
and lands on things; it does not go anywhere on purpose, and the run below says
so in its own numbers.
"""
from cortex import Brain
from cortex.tasks import ScreenTask


def main():
    brain = Brain.demo(n=3000, sim_steps=200, explore=1.0)
    task = ScreenTask(url="https://en.wikipedia.org/wiki/Connectome",
                      allow=("wikipedia.org", "wikimedia.org"),
                      max_steps=60, headless=True)
    with task:
        task.reset()
        for t in range(task.max_steps):
            act = brain.look(task.observe(), focus=task.focus(), window=task.window())
            task.act(act)
            if t % 10 == 0:
                print(f"  [{t:>3}] cursor "
                      f"({task.cursor[0]:6.1f},{task.cursor[1]:6.1f})  {act}")
        print("\n", task.report())


if __name__ == "__main__":
    main()
