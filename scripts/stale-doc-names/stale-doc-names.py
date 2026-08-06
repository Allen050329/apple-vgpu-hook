#!/usr/bin/env python3
"""Find a doc comment that names a symbol this workspace deleted.

`cargo doc`'s intra-doc pass checks the `[`link`]` form and nothing checks the
bare-backtick form, which is what most prose in this tree uses. A name that used
to be a function and is now only a sentence is invisible to every other
instrument here.

The discriminator is git, not spelling: report a backticked identifier that
appears **only** inside comments today and that a commit once removed a
definition of. Everything else backticked in this tree -- log tags, census
fields, wire selectors, Vulkan and kernel names -- fails one half or the other.
"""

import collections
import pathlib
import re
import subprocess
import sys

DOC = re.compile(r"^\s*(?:///|//!|//)(.*)$")
TOK = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`|`(?:[A-Za-z0-9_]+::)+([A-Za-z_][A-Za-z0-9_]*)`")
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DEF = re.compile(r"\b(?:fn|const|struct|enum|trait|type|static|union)\s+([A-Za-z_][A-Za-z0-9_]*)")


def ever_defined(root: pathlib.Path) -> set:
    """Every item name a commit has ever removed from a Rust source file."""
    diff = subprocess.run(
        ["git", "-C", str(root), "log", "-p", "--no-color", "--unified=0", "--", "crates/*.rs"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        m.group(1)
        for line in diff.splitlines()
        if line.startswith("-")
        for m in DEF.finditer(line)
    }


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[2]
    crates = root / "crates"
    in_code = set()
    in_docs = collections.defaultdict(list)
    for path in sorted(crates.rglob("*.rs")):
        for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            doc = DOC.match(line)
            if doc:
                for m in TOK.finditer(doc.group(1)):
                    in_docs[m.group(1) or m.group(2)].append(f"{path.relative_to(root)}:{n}")
            else:
                in_code.update(m.group(0) for m in WORD.finditer(line))

    deleted = ever_defined(root)
    hits = sorted(
        (name, sites)
        for name, sites in in_docs.items()
        if name not in in_code and name in deleted
    )
    for name, sites in hits:
        print(f"{name}")
        for site in sites:
            print(f"    {site}")
    print(f"\n{len(hits)} name(s) referenced only from prose and deleted from the tree.")
    print("Most are deliberate history. Read the tense: a sentence saying a name")
    print("*used to* do something is the record this repo wants kept. A sentence")
    print("saying it *does* something is the rot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
