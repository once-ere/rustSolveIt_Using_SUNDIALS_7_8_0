#!/usr/bin/env python3
"""Checks that drive a real posim. Skipped when no binary is available.

These are the ones that can catch physics moving, so they record for
real and compare against what is committed.
"""

import json
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "src"))

import record_video as rv  # noqa: E402

DOC = json.loads((PACKAGE / "recordings.json").read_text())
BASE = (PACKAGE / DOC["base"]).resolve()
WORKSPACE = (PACKAGE / DOC["workspace"]).resolve() if DOC.get("workspace") else None
HAVE_POSIM = WORKSPACE is not None and rv._binary_in(WORKSPACE) is not None

skip = unittest.skipUnless(
    HAVE_POSIM, f"no built posim under {WORKSPACE} (cargo build --release -p posim)"
)


@skip
class Recordings(unittest.TestCase):
    def test_all_committed_recordings_reproduce_byte_for_byte(self):
        r = subprocess.run(
            [sys.executable, str(PACKAGE / "src" / "record_all.py"),
             "--check", "--workspace", str(WORKSPACE)],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("reproduce byte for byte", r.stdout)


@skip
class Frames(unittest.TestCase):
    """What a recorded frame must carry, checked against a real run."""

    @classmethod
    def setUpClass(cls):
        scene = BASE / "videos/scenes/universal_joint.posim"
        cls.bodies, cls.frames, cls.meta = rv.record(
            scene, frames=4, dt=0.01, workspace=WORKSPACE
        )

    def test_a_frame_carries_the_conserved_quantities(self):
        for k in ("t", "E", "P", "L", "n", "o", "c", "j", "gd"):
            self.assertIn(k, self.frames[0], k)

    def test_the_joint_far_end_is_recorded(self):
        # Three joints: hinge, universal, rod.
        j = self.frames[0]["j"]
        self.assertEqual(len(j), 3)
        for entry in j:
            self.assertEqual(len(entry), 3, "point, axis, point_j")

    def test_shared_point_joints_have_coincident_ends(self):
        # hinge and universal hold ONE point, so no strut is drawn
        for pt, _axis, far in self.frames[0]["j"][:2]:
            self.assertEqual(pt, far)

    def test_a_rod_holds_two_distinct_points(self):
        pt, _axis, far = self.frames[0]["j"][2]
        self.assertNotEqual(pt, far)

    def test_the_recorder_never_integrates(self):
        # Each frame is one `step dt` apart, and the times come from
        # posim, not from multiplying dt in Python.
        ts = [f["t"] for f in self.frames]
        for a, b in zip(ts, ts[1:]):
            self.assertAlmostEqual(b - a, 0.01, places=12)

    def test_the_joints_hold(self):
        for f in self.frames:
            self.assertLess(f["gd"], 1e-5)

    def test_meta_names_the_joints_and_the_method(self):
        self.assertEqual([j["kind"] for j in self.meta["joints"]],
                         ["hinge", "universal", "rod"])
        self.assertEqual([j["rows"] for j in self.meta["joints"]], [5, 4, 1])
        self.assertEqual(self.meta["method"], "Ida")


@skip
class Output(unittest.TestCase):
    def test_the_written_page_is_self_contained(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "v.html"
            r = subprocess.run(
                [sys.executable, str(PACKAGE / "src" / "record_video.py"),
                 str(BASE / "videos/scenes/kepler_ellipse.posim"),
                 "-o", str(out), "--frames", "3", "--dt", "0.02",
                 "--workspace", str(WORKSPACE)],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            html = out.read_text()
            self.assertNotRegex(html, r"""(?:src|href)\s*=\s*["']https?://""")
            self.assertIn("const FRAMES", html)
            self.assertEqual(len(json.loads(
                re.search(r"const FRAMES\s*=\s*(\[.*?\]);", html, re.S).group(1)
            )), 4)  # 3 steps -> 4 frames


if __name__ == "__main__":
    unittest.main(verbosity=2)
