#!/usr/bin/env python3
"""Phase-5: assemble index_of_entities.html from the template + the catalog.

The catalog is inlined as a <script type="application/json"> block, so the
result is one self-contained file that opens from file:// with no network
access, no build step and no dependencies — the same constraint the scene
window in posim/src/scene/scene.html works under.

Stdlib only.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "tools", "app_template.html")
CATALOG = os.path.join(ROOT, "index_data", "catalog.json")
OUT = os.path.join(ROOT, "index_of_entities.html")


def main():
    tpl = open(TEMPLATE, encoding="utf-8").read()
    data = json.load(open(CATALOG, encoding="utf-8"))

    # Minified, and with the two sequences that can break out of a <script>
    # block neutralised. `</script` is the real one; `<!--` is the historical
    # one, and both are cheap to rule out.
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    blob = blob.replace("</", "<\\/").replace("<!--", "<\\!--")

    if "/*__CATALOG__*/" not in tpl:
        sys.exit("template is missing the /*__CATALOG__*/ placeholder")
    html = "<!doctype html>\n<html lang=\"en\">\n" + tpl.replace("/*__CATALOG__*/", blob) \
        + "\n</html>\n"

    open(OUT, "w", encoding="utf-8").write(html)
    entries = [e for e in data if not e.get("_meta")]
    print("%s  —  %.0f KB, %d entries, %d examples"
          % (OUT, len(html) / 1024, len(entries),
             sum(len(e["examples"]) for e in entries)))

    # Tier C ships as a SECOND script the page injects on demand. It is a
    # script rather than a fetch on purpose: a `file://` page cannot fetch a
    # sibling file — Chrome treats every local file as its own opaque origin
    # and blocks it — but it can always load one as a <script>. So the payload
    # assigns a global instead of returning JSON.
    cpath = os.path.join(ROOT, "index_data", "catalog_c.json")
    if os.path.exists(cpath):
        cdata = json.load(open(cpath, encoding="utf-8"))
        blob = json.dumps(cdata, separators=(",", ":"), ensure_ascii=False)
        js = ("/* Tier C — the vendored sundials_rs workspace. Loaded on demand by\n"
              "   index_of_entities.html; see its status page for why these entries\n"
              "   carry status \"reference\" rather than an invented snippet. */\n"
              "window.__TIER_C__ = " + blob + ";\n"
              "if (window.__onTierC) window.__onTierC();\n")
        out_c = os.path.join(ROOT, "catalog-c.js")
        open(out_c, "w", encoding="utf-8").write(js)
        print("%s  —  %.0f KB, %d entries (lazy)"
              % (out_c, len(js) / 1024, len(cdata)))


if __name__ == "__main__":
    main()
