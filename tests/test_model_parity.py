"""Dynamic-model parity: the MPC model must track the real robot.

`agimus_franka_description` (the description used by the real Franka) is the
single source of truth for the arm's mass/inertia, joint limits, torque limits,
and joint damping/friction. The MPC optimizes rollouts over
`models/panda_pick_place/panda.xml`, whose inline values can silently drift from
the real robot (and from the MuJoCo sim model in sbmpc_ros). These tests read
the Agimus YAMLs and assert the MPC model still matches.

Torque limits are not asserted here: the MPC applies them at load from the
`_ARM_TORQUE_LIMITS` constant in `panda_pregrasp.py` ([87]*4 + [12]*3), which
matches the Agimus `joint_limits.yaml` effort values; importing that module pulls
in JAX, so it is left out of this lightweight XML parity check.

Known, intentional sim-only deviation (NOT checked here): `armature=0.1`, a
MuJoCo rotor-inertia surrogate the FER URDF does not model.

Skips when `agimus_franka_description` is not on the ROS prefix path (e.g. a bare
pixi shell without the deps overlay sourced).
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

PANDA_XML = Path(__file__).resolve().parents[1] / "models" / "panda_pick_place" / "panda.xml"

# MuJoCo body name -> Agimus inertials.yaml key.
BODY_TO_AGIMUS = {
    "link0": "link0", "link1": "link1", "link2": "link2", "link3": "link3",
    "link4": "link4", "link5": "link5", "link6": "link6", "link7": "link7",
    "hand": "hand", "left_finger": "leftfinger", "right_finger": "rightfinger",
}
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
DEFAULT_RANGE = (-2.8973, 2.8973)  # MJCF `panda` joint default class.


def _floats(text: str) -> list[float]:
    return [float(token) for token in text.split()]


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


def _agimus_share() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("agimus_franka_description"))
    except Exception:
        pass
    # Fall back to scanning the ROS prefix path / the standard deps install.
    candidates = [
        Path(prefix) / "share" / "agimus_franka_description"
        for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(":")
        if prefix
    ]
    candidates.append(
        Path("/opt/sbmpc_deps_ws/install/agimus_franka_description/share/agimus_franka_description")
    )
    for candidate in candidates:
        if (candidate / "robots" / "fer" / "inertials.yaml").exists():
            return candidate
    pytest.skip("agimus_franka_description not available")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _agimus_inertials(share: Path) -> dict:
    arm = _load_yaml(share / "robots" / "fer" / "inertials.yaml")
    hand = _load_yaml(share / "end_effectors" / "agimus_franka_hand" / "inertials.yaml")
    return {**arm, **hand}


def _mjcf_bodies() -> dict[str, ET.Element]:
    root = ET.parse(PANDA_XML).getroot()
    return {body.get("name"): body for body in root.iter("body")}


def _mjcf_joints() -> dict[str, dict[str, str]]:
    root = ET.parse(PANDA_XML).getroot()
    return {
        joint.get("name"): joint.attrib
        for joint in root.iter("joint")
        if joint.get("name") is not None
    }


def test_panda_xml_inertials_match_agimus_description() -> None:
    agimus = _agimus_inertials(_agimus_share())
    bodies = _mjcf_bodies()

    for body_name, agimus_key in BODY_TO_AGIMUS.items():
        inertial = bodies[body_name].find("inertial")
        assert inertial is not None, f"{body_name} has no <inertial>"
        ref = agimus[agimus_key]

        assert _close(float(inertial.get("mass")), float(ref["mass"])), body_name

        pos = _floats(inertial.get("pos"))
        xyz = _floats(str(ref["origin"]["xyz"]))
        assert all(_close(p, q) for p, q in zip(pos, xyz)), f"{body_name} com"

        ref_inertia = ref["inertia"]
        if inertial.get("fullinertia") is not None:
            ixx, iyy, izz, ixy, ixz, iyz = _floats(inertial.get("fullinertia"))
        else:
            ixx, iyy, izz = _floats(inertial.get("diaginertia"))
            ixy = ixz = iyz = 0.0
        expected = {"xx": ixx, "yy": iyy, "zz": izz, "xy": ixy, "xz": ixz, "yz": iyz}
        for axis, value in expected.items():
            assert _close(value, float(ref_inertia[axis])), f"{body_name} I{axis}"


def test_panda_xml_joint_position_limits_match_agimus_description() -> None:
    limits = _load_yaml(_agimus_share() / "robots" / "fer" / "joint_limits.yaml")
    joints = _mjcf_joints()

    for joint_name in ARM_JOINTS:
        attrib = joints[joint_name]
        lo, hi = _floats(attrib["range"]) if "range" in attrib else DEFAULT_RANGE
        ref = limits[joint_name]["limit"]
        assert _close(lo, float(ref["lower"])), f"{joint_name} lower"
        assert _close(hi, float(ref["upper"])), f"{joint_name} upper"


def test_panda_xml_joints_use_menagerie_damping_standard() -> None:
    # Joint dissipation uses the MuJoCo Menagerie Panda standard
    # (armature=0.1, damping=1, no frictionloss) for numerical stability, matching
    # the sim model. The real joint friction (small viscous + Coulomb, per the
    # De Luca/Gaz identification) is a separate model not yet identified.
    root = ET.parse(PANDA_XML).getroot()
    panda_default = next(d for d in root.iter("default") if d.get("class") == "panda")
    default_joint = panda_default.find("joint")
    assert default_joint is not None
    assert _close(float(default_joint.get("armature")), 0.1)
    assert _close(float(default_joint.get("damping")), 1.0)
    assert default_joint.get("frictionloss") in (None, "0", "0.0")

    joints = _mjcf_joints()
    for joint_name in ARM_JOINTS:
        attrib = joints[joint_name]
        # No per-joint override: inherit the standard from the panda class.
        assert "damping" not in attrib, f"{joint_name} re-adds a damping override"
        assert "frictionloss" not in attrib, f"{joint_name} re-adds frictionloss"
