#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional

# Make sure local modules in the repository directory are importable when
# running this script directly.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import required local modules. Fail fast with a helpful error if they are
# not available so the user can fix the environment (no silent fallbacks).
try:
    import parsing  # type: ignore
except Exception as e:  # pragma: no cover - environment error
    raise SystemExit(
        "ERROR: Required module 'parsing' not found. Did you move files?"
    ) from e

try:
    import graph as graph_module  # type: ignore
except Exception as e:  # pragma: no cover - environment error
    raise SystemExit(
        "ERROR: Required module 'graph' not found. Did you move files?"
    ) from e

try:
    # render helpers are in the `render` package/module
    from render import (  # type: ignore
        try_python_graphviz,
        try_system_dot,
        try_networkx,
        which_command as render_which,
    )
except Exception as e:  # pragma: no cover - environment error
    raise SystemExit(
        "ERROR: Required module 'render' not found. Did you move files?"
    ) from e


def say(*parts) -> None:
    """Simple printing helper."""
    print(" ".join(str(p) for p in parts))


def which_command(*names: str) -> Optional[str]:
    """Return the first available executable name from the provided list."""
    for n in names:
        path = shutil.which(n)
        if path:
            return n
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="callGraph - static call graph generator (modular)"
    )

    parser.add_argument(
        "paths", nargs="*", help="Files or directories to parse"
    )
    parser.add_argument(
        "-l",
        "--language",
        dest="language",
        help="Force language (py, c, cpp, rs, js, ts, tsx, ...)",
    )
    parser.add_argument(
        "-s",
        "--start",
        dest="start",
        help="Function(s) to use as starting point (regex)",
    )
    parser.add_argument(
        "-i",
        "--ignore",
        dest="ignore",
        help="Regex of function names to ignore",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Output filename (dot/png/svg/pdf). If omitted, uses temp dir",
    )
    parser.add_argument(
        "-N",
        "--no-show",
        dest="no_show",
        action="store_true",
        help="Do not display generated image",
    )
    parser.add_argument(
        "-P",
        "--full-path",
        dest="full_path",
        action="store_true",
        help="Do not strip path from node labels",
    )
    parser.add_argument(
        "-c",
        "--write-subset-code",
        dest="write_subset_code",
        help="Write subset source file containing only functions included in graph",
    )
    parser.add_argument(
        "-w",
        "--write-functions",
        dest="write_functions",
        action="store_true",
        help="Write each function to separate file (tempdir)",
    )
    parser.add_argument(
        "-J",
        "--jsn-out",
        dest="jsn_out",
        help="Write JSON representation of call graph to file",
    )
    parser.add_argument(
        "-j",
        "--jsn-in",
        dest="jsn_in",
        help="Read JSON representation from file (skip parsing)",
    )
    parser.add_argument(
        "-Y",
        "--yml-out",
        dest="yml_out",
        help="Write YAML representation of call graph to file",
    )
    parser.add_argument(
        "-y",
        "--yml-in",
        dest="yml_in",
        help="Read YAML representation from file (skip parsing)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-O",
        "--obfuscate",
        dest="obfuscate",
        action="store_true",
        help="Obfuscate function names",
    )
    parser.add_argument(
        "-r",
        "--renderer",
        dest="renderer",
        choices=["auto", "python-graphviz", "system-dot", "networkx"],
        default="auto",
        help="Renderer preference",
    )
    parser.add_argument(
        "-p",
        "--parser",
        dest="parser",
        choices=["auto", "regex", "treesitter"],
        default="auto",
        help="Parser backend preference (treesitter is opt-in)",
    )
    parser.add_argument(
        "--treesitter-bundle",
        dest="treesitter_bundle",
        help="Path to tree-sitter language bundle (.so/.dll/.dylib) for treesitter parsing (optional)",
    )

    args = parser.parse_args(argv)

    if not args.paths and not args.jsn_in and not args.yml_in:
        parser.print_help()
        return 1

    tmpdir = None
    output = args.output
    files: List[str] = []

    # If a precomputed call graph is provided, load it and skip parsing.
    if args.jsn_in:
        with open(args.jsn_in, "r", encoding="utf-8") as fh:
            call_graph = json.load(fh)
    elif args.yml_in:
        try:
            import yaml  # type: ignore
        except Exception:
            raise SystemExit("ERROR: PyYAML not available to read YAML input.")
        with open(args.yml_in, "r", encoding="utf-8") as fh:
            call_graph = yaml.safe_load(fh)
    else:
        # Collect input files. When scanning directories, collect_files requires a language.
        files = parsing.collect_files(args.paths, language=args.language)
        language = args.language or parsing.get_script_type(
            files[0], scripts_only=True
        )
        if not language:
            raise SystemExit(
                "ERROR: language could not be determined. Use --language <language>"
            )

        syntax = parsing.define_syntax(language)

        # Conservative parser selection: use treesitter only when explicitly requested.
        if args.parser == "treesitter":
            try:
                # Optional treesitter wrapper is expected to live under the project.
                # If it is not present we fall back to the regex/AST parser.
                from treesitter_wrapper import TreesitterParser  # type: ignore

                ts_parser = TreesitterParser(
                    bundle_path=args.treesitter_bundle, verbose=args.verbose
                )
                parse = ts_parser.parse_files(files, language)
            except Exception as e:
                say(
                    "WARNING: treesitter parsing requested but unavailable; falling back to regex-based parser:",
                    e,
                )
                parse = parsing.parse_files(files, language, syntax)
        else:
            # Default behavior: regex-based parser (Python uses AST internally).
            parse = parsing.parse_files(files, language, syntax)

        if os.environ.get("DUMP_PARSE"):
            import pprint

            try:
                dump = {
                    "shebang": parse.shebang,
                    "func_definition": parse.func_definition,
                    "func_contents": parse.func_contents,
                    "func_call": parse.func_call,
                }
            except Exception as e:
                dump = {
                    "error": "failed to normalize parse result",
                    "exception": repr(e),
                }
            say("DUMP_PARSE set - parse result dump:")
            pprint.pprint(dump)

        if args.write_functions:
            tmpdir = tempfile.mkdtemp(prefix="call_graph_funcs_")
            for func in parse.func_contents:
                for fpath in parse.func_contents[func]:
                    bfile = os.path.basename(fpath)
                    name, ext = os.path.splitext(bfile)
                    out = os.path.join(tmpdir, f"{name}__{func}{ext}")
                    say("Creating function source file", out)
                    with open(out, "w", encoding="utf-8") as oh:
                        oh.write(parse.func_contents[func][fpath])

        call_graph = graph_module.build_call_graph(
            parse, files, ignore_re=args.ignore, language=language
        )

        if args.write_subset_code:
            subset_file = args.write_subset_code
            say(f"Creating subset source file {subset_file}")
            with open(subset_file, "w", encoding="utf-8") as oh:
                if parse.shebang:
                    oh.write(parse.shebang + "\n")
                main_key = "__MAIN__"
                for file_sub in sorted(call_graph.keys()):
                    file_part, sub_part = file_sub.split(":", 1)
                    if sub_part == main_key:
                        continue
                    contents = parse.func_contents.get(sub_part, {}).get(
                        file_part
                    )
                    if contents:
                        oh.write(contents + "\n")

        if args.jsn_out:
            with open(args.jsn_out, "w", encoding="utf-8") as oh:
                json.dump(call_graph, oh, indent=2)
            say(f"Wrote JSON to {args.jsn_out}")
        if args.yml_out:
            try:
                import yaml  # type: ignore
            except Exception:
                raise SystemExit(
                    "ERROR: PyYAML not available to write YAML output."
                )
            with open(args.yml_out, "w", encoding="utf-8") as oh:
                yaml.safe_dump(call_graph, oh)
            say(f"Wrote YAML to {args.yml_out}")

    # Optional obfuscation
    if args.obfuscate:
        call_graph = graph_module.obfuscate_call_graph(
            call_graph, ignore=["__MAIN__"]
        )

    # Determine initial nodes (starting points)
    if args.start:
        initial_nodes = sorted(
            [
                k
                for k in call_graph.keys()
                if re.search(args.start, k, re.IGNORECASE)
            ]
        )
        if not initial_nodes:
            say(f"ERROR: Could not find any functions matching '{args.start}'")
            return 1
    else:
        initial_nodes = []
        for file_sub in sorted(call_graph.keys()):
            if (
                (not call_graph[file_sub].get("called_by"))
                or file_sub.endswith(":__MAIN__")
                or file_sub.endswith(":main")
            ):
                initial_nodes.append(file_sub)

    if not call_graph:
        say("ERROR: No call graph data to process")
        return 1

    graph_builder = graph_module.GraphBuilder(call_graph, cluster_files=False)
    for n in initial_nodes:
        graph_builder.plot(n)

    if not graph_builder.node:
        say(
            "ERROR: Could not find any matching function calls in your file(s)."
        )
        return 1

    if not output:
        tmpdir = tmpdir or tempfile.mkdtemp(prefix="call_graph_")
        base = os.path.basename(args.paths[0]) if args.paths else "call_graph"
        output = os.path.join(tmpdir, base)
    dot_name = output
    if not dot_name.endswith(".dot"):
        dot_name = os.path.splitext(output)[0] + ".dot"

    graph_builder.generate_dot(
        dot_name, files if "files" in locals() else [], full_path=args.full_path
    )

    # Render to image if requested
    if not output.endswith(".dot"):
        if output.endswith(".svg"):
            fmt = "svg"
        elif output.endswith(".pdf"):
            fmt = "pdf"
        else:
            fmt = "png"
        image = os.path.splitext(output)[0] + "." + fmt
        renderer = getattr(args, "renderer", "auto")
        rendered_ok = False

        if renderer == "auto":
            path = try_python_graphviz(dot_name, output, fmt)
            if path:
                image = path
                rendered_ok = True
            else:
                path = try_system_dot(dot_name, image, fmt)
                if path:
                    image = path
                    rendered_ok = True
                else:
                    path = try_networkx(graph_builder, image)
                    if path:
                        image = path
                        rendered_ok = True
        elif renderer == "python-graphviz":
            path = try_python_graphviz(dot_name, output, fmt)
            if path:
                image = path
                rendered_ok = True
            else:
                say(
                    "ERROR: python 'graphviz' package not available or failed to render as requested by --renderer"
                )
                return 1
        elif renderer == "system-dot":
            path = try_system_dot(dot_name, image, fmt)
            if path:
                image = path
                rendered_ok = True
            else:
                say(
                    "ERROR: system 'dot' not available or failed to render as requested by --renderer"
                )
                return 1
        elif renderer == "networkx":
            path = try_networkx(graph_builder, image)
            if path:
                image = path
                rendered_ok = True
            else:
                say(
                    "ERROR: networkx/matplotlib not available or failed to render as requested by --renderer"
                )
                return 1

        if not rendered_ok:
            say("WARNING: Could not render image - no renderer succeeded.")
            say("Dot file generated at", dot_name)
            return 0

        say("Converting to", image)
        if not args.no_show:
            viewer = render_which(
                "eog",
                "eom",
                "fim",
                "feh",
                "xdg-open",
                "open",
                "display",
                "evince",
            )
            if viewer:
                try:
                    if sys.platform == "win32":
                        subprocess.Popen([viewer, image])
                    else:
                        subprocess.Popen(
                            [viewer, image],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    say(f"Displaying {image} using '{viewer}'")
                except Exception as e:
                    say("WARNING: Could not display image:", e)
            else:
                say("WARNING: Could not find a command to display", image)
    else:
        say("Dot file generated at", dot_name)

    # Verbose: extra heuristic inspection
    if args.verbose and not (args.jsn_in or args.yml_in):
        say(
            "\nVerbose global variable and external script analysis (best-effort):"
        )
        parse_result = parsing.parse_files(
            files,
            args.language or parsing.get_script_type(files[0]),
            parsing.define_syntax(
                args.language or parsing.get_script_type(files[0])
            ),
        )
        sub_info = {}
        for file_sub in sorted(graph_builder.node.keys()):
            file_part, sub_part = file_sub.split(":", 1)
            contents = parse_result.func_contents.get(sub_part, {}).get(
                file_part, ""
            )
            if not contents:
                continue
            vars_used = set()
            scripts_called = set()
            for line in contents.splitlines():
                var_re = parsing.LANG_SYNTAX["variable"].get(
                    args.language or parsing.get_script_type(files[0]), None
                )
                if var_re:
                    for m in re.finditer(var_re, line):
                        v = m.group(1) if m.groups() else m.group(0)
                        if v and not v.isdigit():
                            vars_used.add(v)
                m_script = re.search(
                    r"(\w+\.(?:awk|m|js|php|pl|py|r|rb|sc|sh|tcl)\b)", line
                )
                if m_script:
                    scripts_called.add(m_script.group(1))
            sub_info.setdefault(file_part, {})[sub_part] = {
                "vars": sorted(vars_used),
                "scripts": sorted(scripts_called),
            }
        for f in sorted(sub_info.keys()):
            say("  ", f)
            for s in sorted(sub_info[f].keys()):
                say("    ", s)
                for v in sub_info[f][s]["vars"]:
                    say("      var:", v)
                for sc in sub_info[f][s]["scripts"]:
                    say("      script:", sc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
