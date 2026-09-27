"""Test offscreen rendering and frame extraction for CI visual verification."""
import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.go1_env import go1_env


def test_rgb_array_render():
    """Verify env can render offscreen rgb_array frames without crash."""
    env = go1_env(render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    frames = []

    for _ in range(10):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, done, truncated, _ = env.step(action)
        frame = env.render()
        assert frame is not None, "env.render() returned None in rgb_array mode"
        assert isinstance(frame, np.ndarray), f"Expected ndarray frame, got {type(frame)}"
        assert frame.ndim == 3 and frame.shape[2] == 3, f"Unexpected frame shape {frame.shape}"
        assert frame.dtype == np.uint8, f"Unexpected dtype {frame.dtype}"
        frames.append(frame)
        if done or truncated:
            break

    env.close()
    assert len(frames) >= 5, f"Expected at least 5 frames, got {len(frames)}"
    # Verify non-trivial frame (standard deviation > 0, not all black/white)
    assert np.std(frames[0]) > 5.0, "Rendered frame appears uniformly blank"


def test_headless_gif_generation(tmp_path=None):
    """Verify multi-frame GIF generation via imageio works in headless mode."""
    import imageio

    env = go1_env(render_mode="rgb_array", cpg_mode="fixed_residual")
    obs, _ = env.reset(seed=123)
    frames = []

    for _ in range(15):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, done, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if done or truncated:
            break
    env.close()

    if tmp_path is not None:
        gif_file = os.path.join(str(tmp_path), "ci_smoke.gif")
    else:
        gif_file = os.path.join(REPO_ROOT, "gifs", "ci_smoke_test.gif")
        os.makedirs(os.path.dirname(gif_file), exist_ok=True)

    imageio.mimsave(gif_file, frames, fps=15)
    assert os.path.isfile(gif_file), f"GIF file not created at {gif_file}"
    assert os.path.getsize(gif_file) > 1000, f"GIF file unexpectedly small ({os.path.getsize(gif_file)} bytes)"

    # Clean up test artifact if not in pytest tmp_path
    if tmp_path is None and os.path.exists(gif_file):
        os.remove(gif_file)


if __name__ == "__main__":
    test_rgb_array_render()
    print("PASS test_rgb_array_render")
    test_headless_gif_generation()
    print("PASS test_headless_gif_generation")
    print("All headless render tests passed!")
