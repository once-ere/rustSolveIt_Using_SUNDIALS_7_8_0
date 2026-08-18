#!/usr/bin/env python3
"""Checks that need no posim binary and no network.

Run directly, or through CTest. stdlib unittest only — the recorder has
no third-party dependency and neither do its tests.
"""

import json
import os
import pathlib
import re
import sys
import tempfile
import unittest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "src"))

import record_video as rv  # noqa: E402


def _fake_workspace(root: pathlib.Path, profile="release") -> pathlib.Path:
    """A directory that looks like a built cargo workspace."""
    b = root / "target" / profile
    b.mkdir(parents=True, exist_ok=True)
    exe = b / "posim"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    return root


class SetupLines(unittest.TestCase):
    """Only executable posim reaches the child process."""

    def lines(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "s.posim"
            p.write_text(text)
            return list(rv.setup_lines(p))

    def test_comments_and_blanks_are_dropped(self):
        self.assertEqual(
            self.lines("# a comment\n\n   \nnew sphere as a\n"),
            ["new sphere as a"],
        )

    def test_trailing_comment_is_trimmed(self):
        self.assertEqual(self.lines("collide off  # why\n"), ["collide off"])

    def test_notebook_magics_are_dropped(self):
        # `%...` is handled by the notebook layer; machine mode has no
        # use for it and would only report an error.
        self.assertEqual(self.lines("%time\nstep 0.1\n"), ["step 0.1"])

    def test_scene_lines_are_dropped(self):
        # A recording has no live window to talk to.
        self.assertEqual(
            self.lines("scene create\nnew sphere as a\n"), ["new sphere as a"]
        )


class WorkspaceDiscovery(unittest.TestCase):
    def test_explicit_workspace_wins(self):
        with tempfile.TemporaryDirectory() as d:
            ws = _fake_workspace(pathlib.Path(d) / "ws")
            self.assertEqual(rv.find_workspace(explicit=str(ws)), ws.resolve())

    def test_debug_is_accepted_when_release_is_absent(self):
        with tempfile.TemporaryDirectory() as d:
            ws = _fake_workspace(pathlib.Path(d) / "ws", profile="debug")
            self.assertEqual(rv.find_workspace(explicit=str(ws)), ws.resolve())

    def test_release_is_preferred_over_debug(self):
        with tempfile.TemporaryDirectory() as d:
            ws = pathlib.Path(d) / "ws"
            _fake_workspace(ws, profile="debug")
            _fake_workspace(ws, profile="release")
            self.assertEqual(rv._binary_in(ws).parent.name, "release")

    def test_scene_chooses_its_own_workspace_not_a_sibling(self):
        """Regression: two workspaces in one checkout.

        The recorder used to scan each ancestor's immediate children,
        which resolves to whichever directory name sorts first. In a
        checkout holding a port next to the upstream it came from, that
        is silently the wrong engine — and the recordings that do not
        use newer grammar still come out byte-identical, so nothing
        complains. The scene decides: it lives inside the workspace it
        belongs to.
        """
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            _fake_workspace(root / "AAA_upstream")  # sorts first
            mine = _fake_workspace(root / "zzz_port")
            scene = mine / "videos" / "scenes" / "s.posim"
            scene.parent.mkdir(parents=True)
            scene.write_text("collide off\n")
            os.chdir(root)
            self.assertEqual(rv.find_workspace(near=scene), mine.resolve())

    def test_a_missing_workspace_is_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm:
                rv.find_workspace(explicit=d)
            self.assertIn("cargo build", str(cm.exception))


class Page(unittest.TestCase):
    """The player must work from file:// on a machine with no network."""

    def test_template_fetches_nothing(self):
        for bad in ("http://", "https://", "//cdn", "fetch(", "XMLHttpRequest",
                    "WebSocket", "import(", "@import"):
            self.assertNotIn(bad, rv.PAGE, f"template reaches out via {bad!r}")

    def test_template_has_every_placeholder(self):
        for ph in ("__TITLE__", "__CAPTION__", "__BODIES__", "__FRAMES__",
                   "__META__", "__VIEW__"):
            self.assertIn(ph, rv.PAGE, f"{ph} missing from the template")

    def test_player_reads_the_far_end_of_a_joint(self):
        # A rod's two ends differ and the strut between them is drawn; a
        # ball, hinge or universal joint holds one shared point, so the
        # ends coincide and the line is skipped. Keyed on geometry, not
        # on the joint's name.
        self.assertIn("for (const [pt, axis, far] of f.j)", rv.PAGE)


class Manifest(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads((PACKAGE / "recordings.json").read_text())
        self.base = (PACKAGE / self.doc["base"]).resolve()

    def test_every_scene_and_output_exists(self):
        for e in self.doc["recordings"]:
            self.assertTrue((self.base / e["scene"]).is_file(), e["scene"])
            self.assertTrue((self.base / e["out"]).is_file(), e["out"])

    def test_every_entry_is_complete(self):
        for e in self.doc["recordings"]:
            for k in ("name", "scene", "out", "frames", "dt", "view",
                      "title", "caption"):
                self.assertIn(k, e, f"{e.get('name')} lacks {k}")
            self.assertIn(e["view"], ("iso", "front"))
            self.assertGreater(e["frames"], 0)
            self.assertGreater(e["dt"], 0)

    def test_names_are_unique(self):
        names = [e["name"] for e in self.doc["recordings"]]
        self.assertEqual(len(names), len(set(names)))

    def test_the_workspace_is_pinned(self):
        # Searching for it is what went wrong once already.
        self.assertIn("workspace", self.doc)

    def test_titles_match_what_each_recording_carries(self):
        """The manifest is the authority; drift would silently rename a
        video the next time it is recorded."""
        for e in self.doc["recordings"]:
            html = (self.base / e["out"]).read_text()
            got = re.search(r"<title>(.*?)</title>", html, re.S).group(1).strip()
            self.assertEqual(got, e["title"], e["name"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
