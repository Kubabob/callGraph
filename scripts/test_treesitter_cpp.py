#!/usr/bin/env python3
"""
test_treesitter_cpp.py

Small helper script to exercise the Treesitter POC on C++ files and print a
concise parse summary.

Usage:
  - From the repository root run:
      python callGraph.py/scripts/test_treesitter_cpp.py
    (it will default to the included test/example.cpp)

  - To specify files:
      python callGraph.py/scripts/test_treesitter_cpp.py path/to/foo.cpp path/to/bar.cc

  - To provide a tree-sitter combined bundle:
      python callGraph.py/scripts/test_treesitter_cpp.py --bundle /path/to/liblangs.so ...

Notes:
- This script is conservative: it will print helpful messages if the optional
  py-tree-sitter bindings or language artifacts are not available.
- It does not modify the repository; it only imports and runs the wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import pprint
import sys
from typing import List, Optional

# Ensure the package root (callGraph.py/) is importable so we can import the
# local `treesitter_wrapper` module. This script lives in callGraph.py/scripts/.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def find_default_example() -> Optional[str]:
    candidate = os.path.join(REPO_ROOT, "test", "example.cpp")
    return candidate if os.path.exists(candidate) else None


def summarize_parse_result(parse) -> dict:
    """
    Normalize and summarize the parse result into a plain dict suitable for
    pretty-printing / JSON output.
    The parse object may be either the ParseResult dataclass or a simple object
    with attributes `shebang`, `func_contents`, `func_definition`, `func_call`.
    """
    shebang = getattr(parse, "shebang", None)
    func_contents = getattr(parse, "func_contents", {}) or {}
    func_definition = getattr(parse, "func_definition", {}) or {}
    func_call = getattr(parse, "func_call", {}) or {}

    summary = {"shebang": shebang, "functions": {}}

    for fname in sorted(
        set(
            list(func_contents.keys())
            + list(func_definition.keys())
            + list(func_call.keys())
        )
    ):
        entry = {}
        # definitions (per-file -> line)
        defs = func_definition.get(fname, {})
        entry["definitions"] = defs if defs else {}
        # contents: show which files and the number of lines captured
        cont = {}
        for fpath, txt in func_contents.get(fname, {}).items():
            lines = txt.count("\n") + (
                1 if txt and not txt.endswith("\n") else 0
            )
            cont[fpath] = {"chars": len(txt), "lines": lines}
        entry["contents"] = cont
        # calls: per-file map of called_name -> count
        calls = {}
        for caller, fmap in func_call.items():
            # func_call is keyed by caller; we only want entries where caller == fname
            # but some parsers may use nested structures keyed by caller name
            pass
        # Directly look up func_call entries for this function (it may be the caller)
        calls_map = func_call.get(fname, {}) or {}
        entry["calls"] = calls_map
        summary["functions"][fname] = entry

    # Also include reverse mapping: for each caller show who calls it? This tool
    # keeps it simple and shows calls per caller only (as returned).
    return summary


def print_human(parse):
    print("Tree-sitter parse summary\n" + "-" * 28)
    shebang = getattr(parse, "shebang", None)
    if shebang:
        print("Shebang:", shebang)
    funcs = getattr(parse, "func_contents", {}) or {}
    defs = getattr(parse, "func_definition", {}) or {}
    calls = getattr(parse, "func_call", {}) or {}

    all_names = sorted(
        set(list(funcs.keys()) + list(defs.keys()) + list(calls.keys()))
    )
    if not all_names:
        print("No functions or call sites detected.")
        return

    for name in all_names:
        print(f"\nFunction: {name}")
        # definitions
        d = defs.get(name, {})
        if d:
            for fpath, lineno in d.items():
                print(f"  Defined: {fpath} @ line {lineno}")
        else:
            print("  Defined: (no explicit definition recorded)")

        # content summary
        c = funcs.get(name, {})
        if c:
            for fpath, txt in c.items():
                lines = txt.count("\n") + (
                    1 if txt and not txt.endswith("\n") else 0
                )
                print(f"  Content: {fpath} -> {lines} lines, {len(txt)} chars")
        else:
            print("  Content: (none)")

        # calls made by this function
        call_map = calls.get(name, {}) or {}
        if call_map:
            print("  Calls:")
            for fpath, cmap in call_map.items():
                # cmap: called_name -> count
                for called, cnt in sorted(
                    cmap.items(), key=lambda t: (-t[1], t[0])
                ):
                    print(f"    {called} (in {fpath}) -> {cnt}")
        else:
            print("  Calls: (none detected)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Treesitter C++ POC and print a summary"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="C/C++ files to parse (default: test/example.cpp)",
    )
    parser.add_argument(
        "--bundle",
        "-b",
        dest="bundle",
        help="Path to tree-sitter language bundle (.so/.dll/.dylib)",
        default=None,
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Also print machine-readable JSON summary",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose output from wrapper",
    )
    args = parser.parse_args(argv)

    files = args.paths or []
    if not files:
        default = find_default_example()
        if default:
            files = [default]
        else:
            print(
                "No input files specified and default test/example.cpp not found."
            )
            return 2

    # Verify files exist
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        print("ERROR: The following input files were not found:")
        for m in missing:
            print("  ", m)
        return 2

    # Try to import the TreesitterParser wrapper
    try:
        from treesitter_wrapper import TreesitterParser  # type: ignore
    except Exception as e:
        print("ERROR: Could not import TreesitterParser. Details:")
        print("  ", e)
        print("")
        print(
            "Make sure py-tree-sitter is installed in your environment and/or"
        )
        print(
            "provide a compiled language bundle via --bundle. Falling back is"
        )
        print(
            "possible by re-running without --parser treesitter in the main tool."
        )
        return 3

    # Instantiate and run
    try:
        ts = TreesitterParser(bundle_path=args.bundle, verbose=args.verbose)
    except Exception as e:
        print("ERROR: Failed to instantiate TreesitterParser:")
        print("  ", e)
        return 4

    try:
        parse = ts.parse_files(files, "cpp")
    except Exception as e:
        print("ERROR: Treesitter parsing failed:")
        print("  ", e)
        return 5

    # Print human-friendly summary
    print_human(parse)

    # Optionally print JSON
    if args.as_json:
        summary = summarize_parse_result(parse)
        print("\nJSON summary:")
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
