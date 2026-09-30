"""Test MuJoCo XML assets and model compilation integrity."""
import os
import mujoco

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MENAGERIE_DIR = os.path.join(REPO_ROOT, "mujoco_menagerie", "unitree_go1")
SCENE_XML = os.path.join(MENAGERIE_DIR, "scene.xml")
OBSTACLES_XML = os.path.join(MENAGERIE_DIR, "scene_obstacles.xml")
HURDLE_FLAT_XML = os.path.join(MENAGERIE_DIR, "scene_hurdle_flat.xml")
HURDLE_XML = os.path.join(MENAGERIE_DIR, "scene_hurdle.xml")
GO1_XML = os.path.join(MENAGERIE_DIR, "go1.xml")


def test_scene_xml_compilation():
    """Verify flat scene compiles and contains robot + actuators."""
    assert os.path.isfile(SCENE_XML), f"Missing {SCENE_XML}"
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    assert model.nu == 12, f"Expected 12 actuators, found {model.nu}"
    assert model.nq >= 18, f"Expected >= 18 qpos DOFs, found {model.nq}"
    # Verify feet sites exist
    for site_name in ("FR", "FL", "RR", "RL"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        assert site_id >= 0, f"Missing required site {site_name}"


def test_obstacles_xml_compilation():
    """Verify obstacle scene compiles and contains bumps and hurdle geoms."""
    assert os.path.isfile(OBSTACLES_XML), f"Missing {OBSTACLES_XML}"
    model = mujoco.MjModel.from_xml_path(OBSTACLES_XML)
    assert model.nu == 12
    # Verify hurdle geom
    hurdle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hurdle")
    assert hurdle_id >= 0, "Missing hurdle geom in scene_obstacles.xml"
    # Verify bump geoms
    for bump in ("bump1", "bump2", "bump3"):
        bump_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, bump)
        assert bump_id >= 0, f"Missing bump geom {bump} in scene_obstacles.xml"


def test_hurdle_flat_xml_compilation():
    """Verify hurdle_flat scene contains three bump boxes only (no hurdle box)."""
    assert os.path.isfile(HURDLE_FLAT_XML), f"Missing {HURDLE_FLAT_XML}"
    model = mujoco.MjModel.from_xml_path(HURDLE_FLAT_XML)
    assert model.nu == 12
    # Verify bump geoms present
    for bump in ("bump1", "bump2", "bump3"):
        bump_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, bump)
        assert bump_id >= 0, f"Missing bump geom {bump} in scene_hurdle_flat.xml"
    # Verify hurdle geom absent
    hurdle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hurdle")
    assert hurdle_id < 0, "hurdle geom should NOT be in scene_hurdle_flat.xml"


def test_hurdle_xml_compilation():
    """Verify hurdle scene contains larger hurdle box only (no bump boxes)."""
    assert os.path.isfile(HURDLE_XML), f"Missing {HURDLE_XML}"
    model = mujoco.MjModel.from_xml_path(HURDLE_XML)
    assert model.nu == 12
    # Verify hurdle geom present
    hurdle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hurdle")
    assert hurdle_id >= 0, "Missing hurdle geom in scene_hurdle.xml"
    # Verify bump geoms absent
    for bump in ("bump1", "bump2", "bump3"):
        bump_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, bump)
        assert bump_id < 0, f"Bump geom {bump} should NOT be in scene_hurdle.xml"


def test_go1_xml_compilation():
    """Verify raw go1 robot definition compiles."""
    assert os.path.isfile(GO1_XML), f"Missing {GO1_XML}"
    model = mujoco.MjModel.from_xml_path(GO1_XML)
    assert model.nu == 12
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    assert trunk_id >= 0, "Missing trunk body"


if __name__ == "__main__":
    test_scene_xml_compilation()
    print("PASS test_scene_xml_compilation")
    test_obstacles_xml_compilation()
    print("PASS test_obstacles_xml_compilation")
    test_hurdle_flat_xml_compilation()
    print("PASS test_hurdle_flat_xml_compilation")
    test_hurdle_xml_compilation()
    print("PASS test_hurdle_xml_compilation")
    test_go1_xml_compilation()
    print("PASS test_go1_xml_compilation")
    print("All model asset tests passed!")

