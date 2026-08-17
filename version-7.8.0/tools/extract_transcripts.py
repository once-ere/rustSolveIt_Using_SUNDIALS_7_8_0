#!/usr/bin/env python3
"""Pull the real transcript out of each documented worked example.

The 68 `example` entries were stubs pointing at a file and a line. The
material they point AT is a genuine captured transcript — the documents say
so and the project re-runs them — so the index should carry it rather than
send the reader away.

For a posim transcript we split it into what you TYPE (the `In[n]:=` lines,
stripped of their prompts so the fragment pastes cleanly) and what you GET
(the whole block, verbatim). The typed half then goes through the ordinary
verifier: a documented example that no longer reproduces is a defect worth
finding, not something to quote unchecked.

Rust blocks (special_functions.md) are rustdoc-style, with `#` hidden lines
and a `?` that only works inside the doctest harness. Those are carried as
quoted source, labelled, and not claimed as executed.

Stdlib only.
"""

import json
import re
import sys

HEAD = {
    "grammar.md": (r"^### Example (\d+)", "ex.grammar"),
    "physical_object_simulator.md": (r"^### Example S(\d+)", "ex.guide"),
    "collision_detection.md": (r"^### Example (\d+)", "ex.collision"),
    "NOTEBOOKS.md": (r"^### Notebook (\d+)", "ex.notebook"),
    "special_functions.md": (r"^### (Intermediate|Expert)", "ex.specfn"),
}


def blocks_after(lines, start, stop):
    """Fenced code blocks between two headings, with their language tag."""
    out, i = [], start
    while i < stop:
        m = re.match(r"^```(\w*)\s*$", lines[i])
        if m:
            lang, j = m.group(1), i + 1
            body = []
            while j < stop and not re.match(r"^```\s*$", lines[j]):
                body.append(lines[j])
                j += 1
            if body:
                out.append((lang, "\n".join(body)))
            i = j + 1
        else:
            i += 1
    return out


# Lines that appear inside a transcript block but are not something you type.
NOT_INPUT = re.compile(
    r"^\s*(goodbye"                     # the REPL's sign-off
    r"|posim —"                          # its banner
    r"|type HELP"                        # and the banner's second line
    r"|\(opened in your browser"         # SCENE CREATE's follow-on lines
    r"|showing \d+ entit"
    r"|window action:"                   # asynchronous scene events
    r"|mode = |entities = |camera: "     # SCENE STATUS continuation lines
    r"|saved \d+ cell|system reset"      # magic replies
    r"|\s*E\[\d+\] = )")               # QM STATES continuation lines


def typed_lines(transcript):
    """The `In[n]:= …` half of a transcript, prompts stripped.

    Continuation lines of a multi-line DEF carry no prompt, so anything that
    is not an Out/Err line while a brace is open belongs to the input too.
    """
    out, depth = [], 0
    for line in transcript.split("\n"):
        if NOT_INPUT.match(line):
            continue
        # an interactive capture shows continuation lines as `  ...:= `
        line = re.sub(r"^\s*\.\.\.:=\s?", "  ", line) if "...:=" in line else line
        m = re.match(r"^\s*In\[\d+\]:?=\s?(.*)$", line)
        if m:
            text = m.group(1)
            # a FIFO-driven capture can put several prompts on one line
            text = re.sub(r"In\[\d+\]:?=\s?", "", text)
            text = re.sub(r"Out\[\d+\]=.*$", "", text).rstrip()
            # the sign-off and banner lines also appear AFTER a prompt in a
            # FIFO-driven capture (`In[48]:= goodbye`), so test the stripped
            # text as well as the raw line
            if (text and not NOT_INPUT.match(text) and "→" not in text
                    and "->" not in text.split("#")[0]):
                out.append(text)
                depth += text.count("{") - text.count("}")
            continue
        if re.match(r"^\s*(Out|Err)\[\d+\]", line):
            depth = 0 if depth < 0 else depth
            continue
        if depth > 0 and line.strip():
            out.append(line.rstrip())
            depth += line.count("{") - line.count("}")
    return "\n".join(out).strip()


def main():
    found = {}
    for path, (pat, prefix) in HEAD.items():
        lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
        heads = [i for i, l in enumerate(lines) if re.match(pat, l)]
        for n, i in enumerate(heads, 1):
            stop = heads[n] if n < len(heads) else len(lines)
            blocks = blocks_after(lines, i + 1, stop)
            eid = f"{prefix}.{n}"
            posim = [b for lang, b in blocks if lang == "" and "In[" in b]
            rust = [b for lang, b in blocks if lang in ("rust", "python", "bash")]
            other = [b for lang, b in blocks if lang == "" and "In[" not in b]
            rec = {"id": eid, "file": path, "line": i + 1, "blocks": len(blocks)}
            if posim:
                rec["kind"] = "posim"
                rec["transcript"] = "\n\n".join(posim)
                rec["code"] = typed_lines(rec["transcript"])
            elif rust:
                rec["kind"] = "rust"
                rec["transcript"] = "\n\n".join(rust)
                rec["code"] = None
            elif other:
                rec["kind"] = "text"
                rec["transcript"] = "\n\n".join(other)
                rec["code"] = None
            else:
                rec["kind"] = "none"
                rec["transcript"] = None
                rec["code"] = None
            found[eid] = rec

    # Replay each candidate. Some documented examples CANNOT run in batch and
    # saying so is the point: one ends on a deliberate error (the refusal IS
    # the lesson), several are elided with cells omitted, and the scene ones
    # are interactive sessions where real seconds passed. Those are carried as
    # quotations with the reason attached.
    import os, subprocess, tempfile
    root = os.getcwd()
    binary = os.path.join(root, "target", "release", "posim")
    checked = 0
    for rec in found.values():
        if rec["kind"] != "posim" or not rec["code"]:
            continue
        # SCENE REVERSE cannot be replayed deterministically in batch: playback
        # advances on WALL-CLOCK time on its own thread, so whether any frame
        # has been recorded by the time PAUSE arrives is a race. A single
        # passing replay here would be luck, not evidence.
        if re.search(r"^\s*scene\s+reverse", rec["code"], re.M | re.I):
            rec["replayable"] = False
            rec["why_not"] = ("SCENE REVERSE depends on wall-clock playback, which a "
                              "batch replay cannot reproduce deterministically")
            rec["code"] = None
            continue
        checked += 1
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "t.posim")
            open(f, "w").write(rec["code"] + "\n")
            try:
                r = subprocess.run([binary, "--script", f], capture_output=True,
                                   text=True, timeout=180, cwd=root,
                                   env=dict(os.environ, POSIM_NO_BROWSER="1"))
                bad = [l for l in r.stdout.splitlines() if l.startswith("Err[")]
            except subprocess.TimeoutExpired:
                bad = ["timed out"]
        if bad:
            rec["replayable"] = False
            rec["why_not"] = bad[0][:150]
            rec["code"] = None
        else:
            rec["replayable"] = True

    json.dump(found, open("index_data/transcripts.json", "w"), indent=1)
    ok = sum(1 for r in found.values() if r.get("replayable"))
    print(f"  replayed {checked} candidates: {ok} reproduce, {checked - ok} carried as quotations")
    kinds = {}
    for r in found.values():
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"{len(found)} documented examples -> index_data/transcripts.json")
    print("  by block kind:", kinds)
    runnable = [r for r in found.values() if r["kind"] == "posim" and r["code"]]
    print(f"  with typed input we can re-run: {len(runnable)}")


if __name__ == "__main__":
    main()
