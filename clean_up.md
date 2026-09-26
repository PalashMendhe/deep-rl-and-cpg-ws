# Clean-Up Log — `/home/plsh/rl_env2`

Scope agreed with the user: **remove only unimportant/derived/duplicated files.**
All training artifacts are kept (they explicitly asked for that): every
`ppo_checkpoint_*.pth`, `gifs/`, `evaluations/`, `models/`, `src/checkpoints/`,
`src/logs/`, and everything under `mujoco_menagerie/`.

Method: **archive-then-verify** — files are `mv`-ed into `_archive/` (reversible),
the test/import/render gates are run, and only then is `_archive/` deleted.

> ⚠️ **Recoverability caveat.** Although a git repo now exists at this path
> (`.git/`, `HEAD -> main`, commit `f44aba8 "first commit"`), it tracks **only
> `README.md`** because `.gitignore` is a bare `*`. Everything cleaned up here was
> untracked, so a hard delete would still have been unrecoverable — which is why
> every removal went through `_archive/` first. Fixing `.gitignore` (see §5.2)
> before the next commit is strongly recommended.

---

## §1 Removed (derived / duplicated / orphaned — ~19.2 MB)

| Item | Size | Reason it is unimportant |
| :--- | :--- | :--- |
| `go1_env.ipynb` | 11 KB | Old env prototype notebook; fully superseded by the live class in `src/go1_env.py`. |
| `eval_render_and_gif_saver.ipynb` | 3 KB | Superseded by `src/eval_render.py` (SB3) and `customs_eval_render.py` (custom PPO). |
| `go1.ipynb` | 33 KB | Early scratch/analysis notebook (20 cells); nothing imports or references it. |
| `ant-v4.ipynb` | 11 MB | Ant-v4 reference notebook with embedded outputs; unrelated to the Go1 project and re-obtainable. |
| `ant_rollout.gif` | 7.9 MB | Ant-v4 rollout render; unrelated to Go1. |
| `go1.xml` (root copy) | 11 KB | Orphan duplicate. `scene.xml` / `scene_obstacles.xml` use `<include file="go1.xml"/>`, which resolves **inside** `mujoco_menagerie/unitree_go1/`. The root copy is never loaded and lacks the 4 `*_Touch` sensors the live copy has. |
| `check.py` | 758 B | Broken scratch script: calls `make_go1_env`, which is undefined in that file. Superseded by `tests/` and `base_ppo.main()`'s random-policy sanity check. |
| `__pycache__/`, `src/__pycache__/`, `tests/__pycache__/`, `.pytest_cache/` | 208 KB | Regenerated automatically on next run. |
| `checkpoints/` (root) | 0 | Empty directory; the real checkpoints are the root `ppo_checkpoint_*.pth` files and `src/checkpoints/your_run_name/`. |

## §2 Kept deliberately (NOT deleted)

Measured at inventory time (before the pass); see §7 for what changed underneath.

* **All `ppo_checkpoint_*.pth`** (then 294 files / 557 MB) — including
  `ppo_checkpoint_latest.pth` (the resume file written by `base_ppo.py`) and
  `ppo_checkpoint_6000640.pth` (a candidate path in `customs_eval_render.py`).
  *Note: these were trained against the pre-fix CPG; see `balance_walk_analysis.md`
  for the retraining recommendation before resuming from them.*
* `gifs/` (then 159 MB) and `evaluations/` (133 MB) — previous rollout evidence.
* `models/rl_model_*.zip` (2.4 MB) — pre-CPG SB3 checkpoints (52-dim obs).
* `src/checkpoints/your_run_name/` and `src/logs/your_run_name/` — SB3 run state.
* `mujoco_menagerie/**` — vendored clone (incl. `unitree_go2/`, docs, CI scripts).
* `src/**`, `tests/**`, `base_ppo.py`, `customs_eval_render.py`,
  `implementation_plan.md`, `balance_walk_analysis.md`, this file.
* Virtualenv: `bin/`, `lib/`, `lib64/`, `share/`, `include/`, `pyvenv.cfg`.

## §3 Verification gates (run after archiving, before deleting `_archive/`)

```bash
python -c "import src.go1_env, src.cpg.hopf, src.terrain.config, base_ppo; print('imports OK')"
python tests/test_cpg.py            # expect 6/6
python tests/test_obstacle_env.py   # expect 9/9
python -c "
from src.go1_env import go1_env
e = go1_env(); o, _ = e.reset(); print('flat obs', o.shape)
e2 = go1_env(xml_file='/home/plsh/rl_env2/mujoco_menagerie/unitree_go1/scene_obstacles.xml')
o2, _ = e2.reset(); print('obstacle obs', o2.shape)"
```

Rollback if any gate fails: `mv _archive/* .` (restores every item to its original path).

---

## §4 Future optional (documented, NOT executed)

These were reviewed and intentionally left alone in this pass. Each is safe to
revisit once retraining is complete:

| Idea | Frees | Note |
| :--- | :--- | :--- |
| Archive all but `ppo_checkpoint_latest.pth` + `ppo_checkpoint_6000640.pth` | ~555 MB | Only those two are referenced by `base_ppo.resolve_checkpoint_path()` and `customs_eval_render.py`. |
| Prune `mujoco_menagerie/unitree_go2/` | 30 MB | No XML/config in this project references Go2. |
| Drop the stale 52-dim SB3 pairs | 2.9 MB | `models/rl_model_*.zip` + `src/checkpoints/your_run_name/` were trained on the old 52-dim obs (`include_ext_obs=False`); `src/eval_render.py` also still hardcodes `your_run_name` / `rl_model_120000_steps.zip`. |
| Clear `src/logs/your_run_name/` | 72 KB | TensorBoard event files from the old run. |

## §5 Known gaps found during cleanup (not fixed here)

1. **No root `requirements.txt`.** The dependency set lives only in the venv and in
   `~/Go1_rl_env/requirements.txt` (`gymnasium[mujoco]==1.3.0`,
   `stable-baselines3[extra]==2.9.0`, `numpy==2.0.2`, `scipy`, `imageio`).
2. **`.gitignore` is a single `*`** (venv-generated template), so nothing is
   version-controlled at this path. A real ignore file would list `lib/`, `bin/`,
   `share/`, `include/`, `__pycache__/`, `.pytest_cache/`, `*.pth`, `gifs/`,
   `evaluations/`, `src/logs/`.
3. **Stale renderer defaults** in `src/eval_render.py` (see §4 row 3).
4. `pytest` is not installed in the venv; the test modules therefore ship a
   `__main__` shim and must be run with the venv interpreter
   (`./bin/python tests/test_cpg.py`), not system `python3`.

## §6 Result

**Pass executed (archive-then-verify):**

1. `mkdir _archive` and `mv` of all §1 items → `_archive/` (19 MB, 10 entries).
2. Gates run against the post-archive tree — all passed:

| Gate | Command | Result |
| :--- | :--- | :--- |
| Imports | `./bin/python -c "import src.go1_env, src.cpg.hopf, src.terrain.config, base_ppo"` | `GATE1 imports OK` |
| CPG unit tests | `./bin/python tests/test_cpg.py` | **6/6 passed** |
| Env integration tests | `./bin/python tests/test_obstacle_env.py` | **9/9 passed** |
| Env smoke | flat + `scene_obstacles.xml` reset | `obs (56,)` both; `flat obs (56,)`, `obstacle obs (56,)` |
| Scene integrity | `MjModel.from_xml_path(scene_obstacles.xml)` | 60 geoms (unchanged) |
| Layout | root `go1.xml` absent, `mujoco_menagerie/unitree_go1/go1.xml` present | as expected |

3. `rm -rf _archive` — 19 MB reclaimed, ~0.2 MB of which was caches.

**Post-pass root listing** (no cleanup targets remain):

```
.git  .gitignore  README.md  balance_walk_analysis.md  base_ppo.py  bin
clean_up.md  customs_eval_render.py  evaluations  gifs  implementation_plan.md
include  lib  lib64  models  mujoco_menagerie  ppo_checkpoint_*.pth  pyvenv.cfg
share  src  tests
```

## §7 External changes observed during the pass (not caused by this cleanup)

The working tree changed underneath this task between the inventory and the
archive/verify step, so the "kept" numbers in §2 are now stale on disk:

| Observation | Evidence |
| :--- | :--- |
| A git repo appeared at the root (was absent at inventory time) | `.git/` created `2026-09-27 03:05:36`, `HEAD -> main`, `origin/main`, commit `f44aba8 "first commit"`, `.git` is 1.4 GB |
| Only `README.md` is tracked | `git ls-files` → 1 entry, because `.gitignore` is a bare `*` (see §5.2) |
| Checkpoints reduced from 294 to **4** | remaining: `ppo_checkpoint_1515520.pth`, `ppo_checkpoint_3010560.pth`, `ppo_checkpoint_6000640.pth`, `ppo_checkpoint_latest.pth` (7.6 MB total) |
| `gifs/` is now **empty** | directory mtime `2026-09-27 03:05:49`; `ls gifs` → 0 files |
| `evaluations/` untouched | 8 files, 133 MB, mtime `2026-07-29` |

This cleanup pass only ever moved its 10 declared items into `_archive/` and then
removed `_archive/`; it did not touch `ppo_checkpoint_*.pth` except the filenames
listed in §1, and never touched `gifs/` or `evaluations/`. Both referenced
retention targets (`ppo_checkpoint_latest.pth` and `ppo_checkpoint_6000640.pth`)
are still present, so `customs_eval_render.py`'s checkpoint resolution still works.

**Note on caches:** `__pycache__/` and `src/__pycache__/` regenerate on every
Python run; they were removed at the end of this pass (after the final validation
re-run, which again reported **6/6** CPG and **9/9** obstacle-env tests passing)
and will reappear as soon as any script or test is executed. That is expected and
not a cleanup regression.

Final tree check after the post-validation cache sweep:
`find . -name '__pycache__' -o -name '.pytest_cache'` (excluding venv and
`mujoco_menagerie`) → **NONE**.



