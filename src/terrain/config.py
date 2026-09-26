"""Terrain / obstacle configuration for Go1 curriculum.

Stage 0 (flat) keeps using scene.xml. Stages 1-2 use scene_obstacles.xml;
this dataclass drives hurdle/bump placement on reset so episode variety
does not require one XML file per episode.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TerrainConfig:
    mode: str = "hurdle"  # flat | rough | hurdle | mixed
    rough_seed: int = 0
    rough_amplitude_m: float = 0.02
    rough_length_m: float = 0.4
    hurdle_x: float = 3.0
    hurdle_height_m: float = 0.12
    hurdle_thickness_m: float = 0.08
    hurdle_width_m: float = 2.0
    randomize_hurdle_x: bool = True
    hurdle_x_range: Tuple[float, float] = (2.5, 5.0)
    bump_names: Tuple[str, ...] = ("bump1", "bump2", "bump3")
    # Rough-stage bump variety (Step 5). Base x-centres match scene_obstacles.xml;
    # each reset jitters them inside ±bump_jitter_m and scales height by a factor
    # drawn from bump_height_range (multiplies the XML base height).
    randomize_bumps: bool = True
    bump_base_x: Tuple[float, ...] = (1.0, 1.5, 2.0)
    bump_jitter_m: float = 0.25
    bump_height_range: Tuple[float, float] = (0.5, 1.5)

    def sample_hurdle_x(self, rng) -> float:
        """Draw a hurdle x-position for this episode (seeded rng)."""
        if not self.randomize_hurdle_x:
            return float(self.hurdle_x)
        lo, hi = self.hurdle_x_range
        return float(rng.uniform(lo, hi))

    def sample_bump_poses(self, rng):
        """Draw (x, height_scale) per bump for rough/mixed stages.

        Returns a list of (name, x, height_scale). When randomize_bumps is
        False the XML defaults are returned (scale 1.0).
        """
        out = []
        for i, name in enumerate(self.bump_names):
            base_x = float(self.bump_base_x[i]) if i < len(self.bump_base_x) else float(1.0 + 0.5 * i)
            if not self.randomize_bumps:
                out.append((name, base_x, 1.0))
                continue
            x = float(rng.uniform(base_x - self.bump_jitter_m, base_x + self.bump_jitter_m))
            lo, hi = self.bump_height_range
            s = float(rng.uniform(lo, hi))
            out.append((name, x, s))
        return out

    def sample_terrain(self, rng, stage: int):
        """Plan Step-5 helper: map curriculum stage -> (mode, hurdle_x, bumps).

        stage 0 -> flat (no obstacles), 1 -> rough (bumps only),
        2 -> hurdle (bumps + hurdle). Returns a dict so training-loop code
        can stay readable without touching MuJoCo internals.
        """
        if stage <= 0:
            return {"mode": "flat", "hurdle_x": None, "bumps": []}
        if stage == 1:
            return {"mode": "rough", "hurdle_x": None,
                    "bumps": self.sample_bump_poses(rng)}
        return {"mode": "hurdle", "hurdle_x": self.sample_hurdle_x(rng),
                "bumps": self.sample_bump_poses(rng)}

    @property
    def hurdle_top_z(self) -> float:
        return float(self.hurdle_height_m)

    def hurdle_geom_size(self):
        """MuJoCo box half-extents (x, y, z)."""
        return (
            float(self.hurdle_thickness_m) / 2.0,
            float(self.hurdle_width_m) / 2.0,
            float(self.hurdle_height_m) / 2.0,
        )

    def hurdle_geom_pos(self, x: float):
        """MuJoCo box center so the hurdle sits on the floor."""
        return (float(x), 0.0, float(self.hurdle_height_m) / 2.0)
