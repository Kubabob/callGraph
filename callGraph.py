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
import ast
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import graphviz as _graphviz  # type: ignore
    from graphviz import Digraph  # type: ignore
except Exception:
    _graphviz = None
    Digraph = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def say(*parts) -> None:
    print(" ".join(str(p) for p in parts))


def which_command(*names: str) -> Optional[str]:
    for n in names:
        path = shutil.which(n)
        if path:
            return n
    return None


# -----------------------------------------------------------------------------
# Language syntax (regex) definitions
# -----------------------------------------------------------------------------
LANG_SYNTAX = {
    # Heuristic regexes for many languages. These are intentionally
    # conservative — regex-based parsing is a fallback and will not be
    # perfect for every language syntax.
    "functionDefinition": {
        "py": r"(\s*)(def)\s+([A-Za-z_]\w+)\s*\(",
        "js": r"(\s*)(?:async\s+)?(?:function)\s+([A-Za-z_]\w+)\s*\(",
        "jsx": r"(\s*)(?:async\s+)?(?:function)\s+([A-Za-z_]\w+)\s*\(",
        "ts": r"(\s*)(?:export\s+)?(?:async\s+)?(?:function|const|let|var)\s+([A-Za-z_]\w+)\s*\(",
        "tsx": r"(\s*)(?:export\s+)?(?:async\s+)?(?:function|const|let|var)\s+([A-Za-z_]\w+)\s*\(",
        "c": r"(\s*)(?=.*\b(?:void|bool|int|short|long|double|char|size_t|float)\b)([^(;]+)\s+([A-Za-z_]\w+)\s*\(",
        "cpp": r"(\s*)(?=.*\b(?:void|bool|int|short|long|double|char|size_t|string|vector|unsigned)\b)([^(;]+)\s+([A-Za-z_]\w+)\s*\(",
        "java": r"(\s*)(?=public|private|protected|static|final|synchronized|abstract)([^(;]+)\s+([A-Za-z_]\w+)\s*\(",
        "rs": r"(\s*)(?:pub\s+|async\s+)?(?:fn)\s+([A-Za-z_]\w+)\b",
        "go": r"(\s*)(?:func)\s+([A-Za-z_]\w+)\b",
        "swift": r"(\s*)(?:func)\s+([A-Za-z_]\w+)\b",
        "rb": r"(\s*)(?:def)\s+([A-Za-z_]\w+)",
        "pl": r"(\s*)(?:sub)\s+([A-Za-z_]\w+)",
        "php": r"(\s*)(?:function)\s+([A-Za-z_]\w+)\s*\(",
        "lua": r"(\s*)(?:function)\s+([A-Za-z_]\w+)",
        "kt": r"(\s*)(?:fun)\s+([A-Za-z_]\w+)",
        "dart": r"(\s*)(?:[A-Za-z_][\w<>]*\s+)?([A-Za-z_]\w+)\s*\(",
        "jl": r"(\s*)(?:function|macro)\s+([A-Za-z_]\w+)",
        "m": r"(\s*)(?:[-+]\s*\(|[A-Za-z_]\w+\s*\()",
        "r": r"(\s*)([A-Za-z_]\w+)\s*\<\-|\s*function\s*\(",
        "sh": r"(\s*)(?:function\s+)?([A-Za-z_]\w+)\s*\(\)?\s*\{",
        "bash": r"(\s*)(?:function\s+)?([A-Za-z_]\w+)\s*\(\)?\s*\{",
        "sc": r"(\s*)(?:def)\s+([A-Za-z_]\w+)\s*\(",
        "pas": r"(\s*)(?:procedure|function)\s+([A-Za-z_]\w+)",
        "v": r"(\s*)(?:fn)\s+([A-Za-z_]\w+)",
    },
    "functionEnd": {
        # Python functions end by dedent; keep a permissive pattern for
        # function body continuation. For block languages use closing brace.
        "py": r"\s*\S",
        "c": r"\s*}",
        "cpp": r"\s*}",
        "rs": r"\s*}",
        "java": r"\s*}",
        "ts": r"\s*}",
        "tsx": r"\s*}",
        "js": r"\s*}",
        "jsx": r"\s*}",
        "go": r"\s*}",
        "swift": r"\s*}",
        "php": r"\s*}",
        "lua": r"\s*end",
        "rb": r"\s*end",
        "pl": r"\s*}",
    },
    "functionCall": {
        # Simple name( pattern for many languages; will also catch some
        # non-call constructs but works as a lightweight heuristic.
        "py": r"([A-Za-z_]\w+)\s*\(",
        "js": r"([A-Za-z_]\w+)\s*\(",
        "jsx": r"([A-Za-z_]\w+)\s*\(",
        "ts": r"([A-Za-z_]\w+)\s*\(",
        "tsx": r"([A-Za-z_]\w+)\s*\(",
        "c": r"([A-Za-z_]\w+)\s*\(",
        "cpp": r"([A-Za-z_]\w+)\s*\(",
        "rs": r"([A-Za-z_]\w+)\s*\(",
        "java": r"([A-Za-z_]\w+)\s*\(",
        "go": r"([A-Za-z_]\w+)\s*\(",
        "rb": r"([A-Za-z_]\w+)\s*\(",
        "php": r"([A-Za-z_]\w+)\s*\(",
        "lua": r"([A-Za-z_]\w+)\s*\(",
    },
    "comment": {
        # Common single-line comment tokens
        "py": r"#",
        "rb": r"#",
        "pl": r"#",
        "sh": r"#",
        "bash": r"#",
        "js": r"//",
        "jsx": r"//",
        "ts": r"//",
        "tsx": r"//",
        "c": r"//",
        "cpp": r"//",
        "rs": r"//",
        "java": r"//",
        "php": r"//",
        "lua": r"--",
    },
    "variable": {
        # Very permissive variable name capture used by the "verbose" analysis.
        "py": r"([A-Za-z_]\w+)",
        "js": r"([A-Za-z_]\w+)",
        "ts": r"([A-Za-z_]\w+)",
        "rs": r"([A-Za-z_]\w+)",
        "rb": r"([A-Za-z_]\w+)",
        "php": r"([A-Za-z_]\w+)",
    },
}


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class ParseResult:
    shebang: Optional[str]
    func_contents: Dict[str, Dict[str, str]]  # func -> file -> contents
    func_definition: Dict[str, Dict[str, int]]  # func -> file -> first line
    func_call: Dict[
        str, Dict[str, Dict[str, int]]
    ]  # caller_func -> file -> called_name -> count


# -----------------------------------------------------------------------------
# NetworkX fallback renderer
# -----------------------------------------------------------------------------
def _render_with_networkx(graph_builder, output_path: str) -> bool:
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except Exception:
        return False

    try:
        G = nx.DiGraph()
        labels = {}
        node_colors = []
        for n in sorted(graph_builder.node.keys()):
            G.add_node(n)
            file_part, sub_part = n.split(":", 1)
            labels[n] = f"{os.path.basename(file_part)}\n{sub_part}"
            node_colors.append(
                "#78a2c8"
                if n not in getattr(graph_builder, "initial_node", {})
                else "#66c2a5"
            )

        for fro in sorted(graph_builder.edge.keys()):
            for to in sorted(graph_builder.edge[fro].keys()):
                G.add_edge(fro, to)

        try:
            pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
        except Exception:
            pos = nx.random_layout(G)

        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200)
        nx.draw_networkx_edges(
            G, pos, arrows=True, arrowstyle="->", arrowsize=12
        )
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        plt.axis("off")

        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".svg":
            plt.savefig(output_path, format="svg", bbox_inches="tight")
        elif ext == ".pdf":
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(output_path, format="png", bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Core parsing
# -----------------------------------------------------------------------------
def define_syntax(language: str):
    # Build compiled syntax dict for a language
    compiled = {}
    for key in (
        "functionDefinition",
        "functionEnd",
        "functionCall",
        "comment",
        "variable",
    ):
        pat = LANG_SYNTAX.get(key, {}).get(language)
        compiled[key] = re.compile(pat, re.IGNORECASE) if pat else None
    return compiled


def parse_files(
    files: List[str], language: str, language_syntax
) -> ParseResult:
    """
    Parse files and produce function definitions, contents and call counts.

    For Python we use the AST. For others we fall back to a line-oriented
    regex-based heuristic. Returned structures are plain dicts (no defaultdicts)
    to make type-checkers and downstream code happy.
    """
    main = "__MAIN__"
    # use temporary nested dicts while parsing
    func_contents: Dict[str, Dict[str, str]] = defaultdict(
        dict
    )  # func -> file -> contents
    func_definition: Dict[str, Dict[str, int]] = defaultdict(
        dict
    )  # func -> file -> first-line
    # nested mapping: caller_func -> file -> called_name -> count
    func_call_nested = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    shebang: Optional[str] = None

    fd_re = language_syntax.get("functionDefinition")
    fe_re = language_syntax.get("functionEnd")
    fc_re = language_syntax.get("functionCall")
    comment_re = language_syntax.get("comment")

    for file in files:
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except Exception:
            text = ""

        if text:
            first_line = text.splitlines()[0] if text.splitlines() else ""
            if first_line.startswith("#!"):
                shebang = first_line.strip()

        # Python: prefer AST for correctness
        if language == "py":
            try:
                tree = ast.parse(text, filename=file)
            except Exception:
                tree = None

            if tree is not None:
                lines = text.splitlines()

                def source_for(node: ast.AST) -> str:
                    start = getattr(node, "lineno", None)
                    end = getattr(node, "end_lineno", None)
                    if start and end:
                        return "\n".join(lines[start - 1 : end])
                    if start:
                        return lines[start - 1]
                    return ""

                class PyVisitor(ast.NodeVisitor):
                    def __init__(self):
                        self.current_stack: List[str] = [main]

                    def visit_FunctionDef(self, node: ast.AST):
                        # Accept ast.AST to avoid type-check conflicts with AsyncFunctionDef
                        name = getattr(node, "name", None)
                        lineno = getattr(node, "lineno", None)
                        if name and lineno:
                            func_definition[name][file] = lineno
                            func_contents[name].setdefault(file, "")
                            func_contents[name][file] += source_for(node) + "\n"
                        self.current_stack.append(name or "<anon>")
                        # traverse body to find nested calls
                        self.generic_visit(node)
                        self.current_stack.pop()

                    def visit_AsyncFunctionDef(self, node: ast.AST):
                        # Handle async defs similarly
                        return self.visit_FunctionDef(node)

                    def visit_ClassDef(self, node: ast.AST):
                        # Do not treat classes as functions; still traverse to find nested calls
                        self.generic_visit(node)

                    def visit_Call(self, node: ast.Call):
                        called = None
                        func_node = node.func
                        if isinstance(func_node, ast.Name):
                            called = func_node.id
                        elif isinstance(func_node, ast.Attribute):
                            called = func_node.attr
                        caller = self.current_stack[-1]
                        if called:
                            func_call_nested[caller][file][called] += 1
                        self.generic_visit(node)

                    def visit_Module(self, node: ast.Module):
                        # Inspect module-level statements as MAIN while ensuring we
                        # still visit every statement so FunctionDef nodes are
                        # processed by their visitor. Only non-def/class top-level
                        # stmts are appended to the MAIN contents.
                        for stmt in node.body:
                            if not isinstance(
                                stmt,
                                (
                                    ast.FunctionDef,
                                    ast.AsyncFunctionDef,
                                    ast.ClassDef,
                                ),
                            ):
                                src = source_for(stmt)
                                if src:
                                    func_contents[main].setdefault(file, "")
                                    func_contents[main][file] += src + "\n"
                            # always visit the statement so FunctionDef / ClassDef
                            # handlers run and populate func_definition, func_contents, etc.
                            self.visit(stmt)

                visitor = PyVisitor()
                visitor.visit(tree)
                # move to next file (AST handled)
                continue

        # Fallback legacy parsing for non-Python or failed AST
        norm_text = text.replace("\\\n", "")
        # small normalization for C-like languages
        if language in ("c", "cpp", "java"):
            norm_text = re.sub(r"\)\s*\n\s*{", ") {", norm_text)

        lines = norm_text.splitlines()
        func_stack: List[str] = [main]
        space_stack: List[str] = [""]
        func_contents[main].setdefault(file, "")
        file_line_num = 0
        in_pod = False

        for line in lines:
            file_line_num += 1
            original_line = line

            # perl __END__ / POD handling (best-effort)
            if language == "pl":
                if line.strip() == "__END__":
                    break
                m_tag = re.match(r"^=(\w+)", line)
                if m_tag:
                    tag = m_tag.group(1)
                    in_pod = tag != "cut"
                    continue
                if in_pod:
                    continue

            # append blank lines to current function contents
            if re.match(r"^\s*(#.*)?$", line):
                func_contents[func_stack[-1]].setdefault(file, "")
                func_contents[func_stack[-1]][file] += line + "\n"
                continue

            # strip comments (best-effort)
            if comment_re:
                try:
                    line = re.sub(rf"{comment_re.pattern}.*", "", line)
                except re.error:
                    pass

            # function definition detection
            if fd_re:
                m = fd_re.match(line)
                if m:
                    # Be defensive about the number and order of regex groups.
                    # Some language patterns put the leading indent in group 1 and the
                    # function name in group 2 or 3; others put only the name.
                    groups = [g for g in m.groups() if g is not None]
                    leading_spaces = groups[0] if groups else ""
                    func_name = ""
                    # Prefer the last group that looks like a valid identifier.
                    for g in reversed(groups):
                        try:
                            g_str = str(g)
                        except Exception:
                            continue
                        if re.match(r"^[A-Za-z_]\w*$", g_str):
                            func_name = g_str
                            break
                    # Fallback: try m.group(m.lastindex) if present
                    if not func_name and getattr(m, "lastindex", None):
                        try:
                            cand = m.group(m.lastindex)
                            if cand and re.match(r"^[A-Za-z_]\w*$", str(cand)):
                                func_name = str(cand)
                        except Exception:
                            pass
                    func_name = func_name.split("::")[-1] if func_name else ""
                    if not func_name:
                        func_name = f"<anonymous_{file_line_num}>"
                        say(
                            f"Found anonymous function at {file}:{file_line_num} -> {func_name}"
                        )
                    func_definition[func_name][file] = file_line_num
                    func_contents[func_name].setdefault(file, "")
                    func_contents[func_name][file] += original_line + "\n"
                    if language == "py":
                        func_stack = [main, func_name]
                        space_stack = ["", leading_spaces]
                        func_contents[main].setdefault(file, "")
                    else:
                        if language not in ("jl",) and re.search(
                            r"}\s*(;\s*)?$", line
                        ):
                            # function ended on same line
                            pass
                        else:
                            func_stack.append(func_name)
                            space_stack.append(leading_spaces)
                    continue

            # detect end of function (for languages with explicit end tokens)
            if func_stack[-1] != main and fe_re:
                try:
                    indent_check = (
                        re.match(
                            r"^" + re.escape(space_stack[-1]) + r"\S", line
                        )
                        is not None
                    )
                except re.error:
                    indent_check = True
                if fe_re and re.match(fe_re.pattern, line) and indent_check:
                    func_contents[func_stack[-1]].setdefault(file, "")
                    func_contents[func_stack[-1]][file] += original_line + "\n"
                    if space_stack[-1] == "":
                        func_stack = [main]
                        space_stack = [""]
                    else:
                        func_stack.pop()
                        space_stack.pop()
                    continue

            # append to current function content
            func_contents[func_stack[-1]].setdefault(file, "")
            func_contents[func_stack[-1]][file] += original_line + "\n"

            # find candidate calls
            if fc_re:
                for mcall in fc_re.finditer(line):
                    called = mcall.group(1)
                    if language in ("tcl", "pl"):
                        called = called.split("::")[-1]
                    if called:
                        func_call_nested[func_stack[-1]][file][called] += 1

    # Normalize nested defaultdicts into plain dicts before returning
    func_contents_out: Dict[str, Dict[str, str]] = {}
    for fn, fmap in func_contents.items():
        func_contents_out[fn] = dict(fmap)

    func_definition_out: Dict[str, Dict[str, int]] = {}
    for fn, fmap in func_definition.items():
        func_definition_out[fn] = dict(fmap)

    func_call_out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for caller, fmap in func_call_nested.items():
        func_call_out[caller] = {}
        for fpath, calls in fmap.items():
            func_call_out[caller][fpath] = dict(calls)

    return ParseResult(
        shebang=shebang,
        func_contents=func_contents_out,
        func_definition=func_definition_out,
        func_call=func_call_out,
    )


# -----------------------------------------------------------------------------
# Build call graph
# -----------------------------------------------------------------------------
def build_call_graph(
    parse: ParseResult,
    files: List[str],
    ignore_re: Optional[str] = None,
    language: Optional[str] = None,
):
    call_graph = defaultdict(lambda: {"calls": {}, "called_by": {}})

    for caller_sub, file_map in parse.func_call.items():
        if ignore_re and re.search(ignore_re, caller_sub):
            continue
        for caller_file, calls in file_map.items():
            for referenced_sub, count in calls.items():
                if ignore_re and re.search(ignore_re, referenced_sub):
                    continue
                if referenced_sub not in parse.func_definition:
                    continue
                caller_key = f"{caller_file}:{caller_sub}"
                # prefer same-file
                if caller_file in parse.func_definition.get(referenced_sub, {}):
                    referenced_key = f"{caller_file}:{referenced_sub}"
                else:
                    referenced_files = sorted(
                        parse.func_definition[referenced_sub].keys()
                    )
                    if len(referenced_files) == 1:
                        referenced_key = (
                            f"{referenced_files[0]}:{referenced_sub}"
                        )
                    else:
                        # ambiguous
                        continue

                call_graph[caller_key]["calls"][referenced_key] = (
                    call_graph[caller_key]["calls"].get(referenced_key, 0)
                    + count
                )
                call_graph[referenced_key]["called_by"][caller_key] = (
                    call_graph[referenced_key]["called_by"].get(caller_key, 0)
                    + 1
                )

    # Conservative rescan by content (skip for python)
    if language != "py":
        for caller_sub, file_map in parse.func_contents.items():
            if ignore_re and re.search(ignore_re, caller_sub):
                continue
            for caller_file, body in file_map.items():
                for line in body.splitlines():
                    for m in re.finditer(r"(\w+)\s*\(", line):
                        referenced_sub = m.group(1)
                        if not referenced_sub:
                            continue
                        if ignore_re and re.search(ignore_re, referenced_sub):
                            continue
                        if referenced_sub not in parse.func_definition:
                            continue
                        if caller_file in parse.func_definition.get(
                            referenced_sub, {}
                        ):
                            referenced_key = f"{caller_file}:{referenced_sub}"
                        else:
                            referenced_files = sorted(
                                parse.func_definition[referenced_sub].keys()
                            )
                            if len(referenced_files) == 1:
                                referenced_key = (
                                    f"{referenced_files[0]}:{referenced_sub}"
                                )
                            else:
                                continue
                        caller_key = f"{caller_file}:{caller_sub}"
                        call_graph.setdefault(
                            caller_key, {"calls": {}, "called_by": {}}
                        )
                        call_graph.setdefault(
                            referenced_key, {"calls": {}, "called_by": {}}
                        )
                        if (
                            referenced_key
                            not in call_graph[caller_key]["calls"]
                        ):
                            call_graph[caller_key]["calls"][referenced_key] = 1
                            call_graph[referenced_key]["called_by"][
                                caller_key
                            ] = 1

    # Range-based rescan (skip for python)
    if language != "py":
        func_starts_by_file: Dict[str, List[tuple]] = defaultdict(list)
        for fname, locations in parse.func_definition.items():
            for fpath, start_line in locations.items():
                func_starts_by_file[fpath].append((start_line, fname))

        for fpath, starts in func_starts_by_file.items():
            starts.sort(key=lambda x: x[0])
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    file_lines = fh.read().splitlines()
            except Exception:
                continue
            nlines = len(file_lines)
            for idx, (start_line, funcname) in enumerate(starts):
                end_line = None
                contents = parse.func_contents.get(funcname, {}).get(fpath)
                if contents is not None:
                    count_lines = len(contents.splitlines())
                    if count_lines > 0:
                        end_line = start_line + count_lines - 1
                if end_line is None:
                    if idx + 1 < len(starts):
                        end_line = starts[idx + 1][0] - 1
                    else:
                        end_line = nlines
                if idx + 1 < len(starts):
                    next_start = starts[idx + 1][0]
                    if end_line >= next_start:
                        end_line = next_start - 1
                if end_line < start_line:
                    continue
                body_lines = file_lines[start_line - 1 : end_line]
                for line in body_lines:
                    for m in re.finditer(r"(\w+)\s*\(", line):
                        callee = m.group(1)
                        if not callee:
                            continue
                        if ignore_re and re.search(ignore_re, callee):
                            continue
                        if callee not in parse.func_definition:
                            continue
                        if fpath in parse.func_definition.get(callee, {}):
                            callee_key = f"{fpath}:{callee}"
                        else:
                            callee_files = sorted(
                                parse.func_definition[callee].keys()
                            )
                            if len(callee_files) == 1:
                                callee_key = f"{callee_files[0]}:{callee}"
                            else:
                                continue
                        caller_key = f"{fpath}:{funcname}"
                        call_graph.setdefault(
                            caller_key, {"calls": {}, "called_by": {}}
                        )
                        call_graph.setdefault(
                            callee_key, {"calls": {}, "called_by": {}}
                        )
                        if callee_key not in call_graph[caller_key]["calls"]:
                            call_graph[caller_key]["calls"][callee_key] = 1
                            call_graph[callee_key]["called_by"][caller_key] = 1

    # remove spurious self-edges added by rescans unless originally present
    for caller_key in list(call_graph.keys()):
        try:
            caller_file, caller_func = caller_key.rsplit(":", 1)
        except ValueError:
            continue
        if caller_key in call_graph.get(caller_key, {}).get("calls", {}):
            has_self_in_parse = False
            if (
                caller_func in parse.func_call
                and caller_file in parse.func_call.get(caller_func, {})
            ):
                if (
                    parse.func_call[caller_func][caller_file].get(
                        caller_func, 0
                    )
                    > 0
                ):
                    has_self_in_parse = True
            if not has_self_in_parse:
                call_graph[caller_key]["calls"].pop(caller_key, None)
                call_graph[caller_key]["called_by"].pop(caller_key, None)

    return dict(call_graph)


# -----------------------------------------------------------------------------
# Graph selection and DOT generation
# -----------------------------------------------------------------------------
class GraphBuilder:
    def __init__(
        self, call_graph: Dict[str, Dict], cluster_files: bool = False
    ):
        self.call_graph = call_graph
        self.node: Dict[str, int] = {}
        self.edge: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.initial_node: Dict[str, int] = {}
        self.cluster_files = cluster_files
        self.clusters: Dict[str, Dict] = {}

    def plot(self, from_file_sub: str, direction: Optional[str] = None):
        direction = direction or "up down"
        self.node[from_file_sub] = self.node.get(from_file_sub, 0) + 1
        if direction == "up down":
            self.initial_node[from_file_sub] = (
                self.initial_node.get(from_file_sub, 0) + 1
            )
            direction = "up down"
        if "up" in direction:
            for parent in sorted(
                self.call_graph.get(from_file_sub, {})
                .get("called_by", {})
                .keys()
            ):
                self.edge[parent][from_file_sub] += 1
                if parent not in self.node:
                    self.plot(parent, "up")
        if "down" in direction:
            for to in sorted(
                self.call_graph.get(from_file_sub, {}).get("calls", {}).keys()
            ):
                self.edge[from_file_sub][to] += 1
                if to not in self.node:
                    self.plot(to, "down")

    def _sanitize_node_id(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", name)

    def generate_dot(
        self, dot_name: str, files: List[str], full_path: bool = False
    ):
        nodes = sorted(self.node.keys())
        id_map = {n: self._sanitize_node_id(n) for n in nodes}

        if Digraph:
            g = Digraph(name="call_graph", format="dot")
            g.attr(rankdir="LR", concentrate="true", ratio="0.7", fontsize="24")
            for node_key in nodes:
                nid = id_map[node_key]
                file, sub = node_key.split(":", 1)
                label_file = os.path.basename(file) if not full_path else file
                display_sub = sub if sub else "<anonymous>"
                label = f"{label_file}\\n{display_sub}"
                attrs = {"label": label}
                if node_key in self.initial_node:
                    attrs["style"] = "filled"
                    attrs["fillcolor"] = "/greens3/2"
                    attrs["color"] = "/greens3/3"
                g.node(nid, **attrs)
            for fro in sorted(self.edge.keys()):
                for to in sorted(self.edge[fro].keys()):
                    g.edge(
                        id_map.get(fro, self._sanitize_node_id(fro)),
                        id_map.get(to, self._sanitize_node_id(to)),
                    )
            with open(dot_name, "w", encoding="utf-8") as fh:
                fh.write(g.source)
            say("Generating:", dot_name)
        else:
            lines = []
            lines.append("digraph call_graph {")
            lines.append("  rankdir=LR;")
            lines.append("  concentrate=true;")
            lines.append("  ratio=0.7;")
            lines.append("  fontsize=24;")
            for node_key in nodes:
                nid = id_map[node_key]
                file, sub = node_key.split(":", 1)
                label_file = os.path.basename(file) if not full_path else file
                display_sub = sub if sub else "<anonymous>"
                label = f"{label_file}\\n{display_sub}"
                attrs = []
                if node_key in self.initial_node:
                    attrs.append("style=filled")
                    attrs.append('fillcolor="/greens3/2"')
                    attrs.append('color="/greens3/3"')
                attrstr = ", ".join(attrs)
                if attrstr:
                    lines.append(f'  "{nid}" [label="{label}", {attrstr}];')
                else:
                    lines.append(f'  "{nid}" [label="{label}"];')
            for fro in sorted(self.edge.keys()):
                for to in sorted(self.edge[fro].keys()):
                    lines.append(
                        f'  "{id_map.get(fro, self._sanitize_node_id(fro))}" -> "{id_map.get(to, self._sanitize_node_id(to))}";'
                    )
            lines.append("}")
            with open(dot_name, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
            say("Generating:", dot_name)


# -----------------------------------------------------------------------------
# Obfuscation (optional)
# -----------------------------------------------------------------------------
def obfuscate_call_graph(
    call_graph: Dict[str, Dict],
    ignore: Optional[List[str]] = None,
    seed: Optional[int] = None,
):
    jargon = [
        "abscond",
        "amplify",
        "assemble",
        "benchmark",
        "calculate",
        "compile",
        "compress",
        "decode",
        "delete",
        "emulate",
        "encode",
        "enhance",
        "generate",
        "inspect",
        "mutate",
        "obfuscate",
        "process",
        "refactor",
        "transform",
        "translate",
        "upload",
        "validate",
    ]
    import random

    if seed is not None:
        random.seed(seed)
    cache: Dict[str, str] = {}
    ignore = ignore or ["__MAIN__"]

    def transform(name: str) -> str:
        if name in cache:
            return cache[name]
        if name in ignore:
            cache[name] = name
            return name
        new = (
            random.choice(jargon)
            if jargon
            else f"func_{random.randint(1000, 9999)}"
        )
        cache[name] = new
        return new

    new_graph: Dict[str, Dict] = {}
    for from_k in call_graph:
        from_file, from_sub = from_k.split(":", 1)
        from_sub_new = transform(from_sub)
        for side in ("calls", "called_by"):
            for to_k in call_graph[from_k].get(side, {}):
                to_file, to_sub = to_k.split(":", 1)
                to_sub_new = transform(to_sub)
                from_new = f"{from_file}:{from_sub_new}"
                to_new = f"{to_file}:{to_sub_new}"
                new_graph.setdefault(from_new, {}).setdefault(side, {})[
                    to_new
                ] = call_graph[from_k][side][to_k]
    return new_graph


# -----------------------------------------------------------------------------
# File collection
# -----------------------------------------------------------------------------
def collect_files(
    paths: List[str], language: Optional[str] = None
) -> List[str]:
    found: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            if not language:
                raise SystemExit(
                    "ERROR: Must specify -language when scanning a directory."
                )
            for root, _, files in os.walk(p):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    lang = get_script_type(
                        fpath, scripts_only=False, forced_language=None
                    )
                    if lang == language:
                        found.append(fpath)
        else:
            found.append(p)
    if not found:
        raise SystemExit("ERROR: No input files found.")
    for f in found:
        if not os.path.exists(f):
            raise SystemExit(f"ERROR: {f} not found!")
    return found


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_script_type(
    file: str, scripts_only: bool = False, forced_language: Optional[str] = None
) -> str:
    """
    Determine a conservative language identifier for a file path.

    - Honors `forced_language` when provided.
    - Looks at file extension; recognizes many common extensions including
      `jsx` and `tsx`.
    - Falls back to shebang detection for scripts.
    - If `scripts_only` is True and no shebang/known ext applies, returns "".
    """
    if forced_language:
        return forced_language
    file = file.split("#", 1)[0]
    ext = file.lower().rsplit(".", 1)
    if len(ext) == 2:
        suffix = ext[1]
        mapping = {
            "py": "py",
            "pyw": "py",
            "js": "js",
            "jsx": "jsx",
            "ts": "ts",
            "tsx": "tsx",
            "c": "c",
            "h": "c",
            "cc": "cpp",
            "cpp": "cpp",
            "cxx": "cpp",
            "hh": "cpp",
            "java": "java",
            "kt": "kt",
            "kts": "kt",
            "go": "go",
            "rs": "rs",
            "rb": "rb",
            "pl": "pl",
            "pm": "pl",
            "php": "php",
            "sh": "sh",
            "bash": "bash",
            "zsh": "sh",
            "lua": "lua",
            "swift": "swift",
            "dart": "dart",
            "jl": "jl",
            "m": "m",
            "r": "r",
            "sc": "sc",
            "scala": "sc",
            "pas": "pas",
            "v": "v",
        }
        if suffix in mapping:
            return mapping[suffix]
    # fallback to shebang sniffing for script files
    if os.path.isfile(file):
        try:
            with open(file, "r", errors="ignore") as fh:
                first = fh.readline().strip()
                m = re.match(r"^#!(?:.*/env\s+)?(\S+)", first)
                if m:
                    fname = os.path.basename(m.group(1))
                    if fname.startswith("python"):
                        return "py"
                    if fname.startswith("perl"):
                        return "pl"
                    if fname == "ruby":
                        return "rb"
                    if fname in ("node", "nodejs", "nodejs.exe"):
                        return "js"
                    if "bash" in fname or "sh" in fname:
                        return "sh"
        except Exception:
            pass
    # final fallback: either empty (for scripts_only) or the extension itself
    return (
        "" if scripts_only else (file.rsplit(".", 1)[-1] if "." in file else "")
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="callGraph - static call graph generator (cleaned)"
    )
    parser.add_argument(
        "paths", nargs="*", help="Files or directories to parse"
    )
    parser.add_argument(
        "-language", help="Force language (pl, py, tcl, js, ...)"
    )
    parser.add_argument(
        "-start", help="Function(s) to use as starting point (regex)"
    )
    parser.add_argument("-ignore", help="Regex of function names to ignore")
    parser.add_argument(
        "-output",
        help="Output filename (dot/png/svg/pdf). If omitted, uses temp dir",
    )
    parser.add_argument(
        "-noShow", action="store_true", help="Do not display generated image"
    )
    parser.add_argument(
        "-fullPath",
        action="store_true",
        help="Do not strip path from node labels",
    )
    parser.add_argument(
        "-writeSubsetCode",
        help="Write subset source file containing only functions included in graph",
    )
    parser.add_argument(
        "-writeFunctions",
        action="store_true",
        help="Write each function to separate file (tempdir)",
    )
    parser.add_argument(
        "-jsnOut", help="Write JSON representation of call graph to file"
    )
    parser.add_argument(
        "-jsnIn", help="Read JSON representation from file (skip parsing)"
    )
    parser.add_argument(
        "-ymlOut", help="Write YAML representation of call graph to file"
    )
    parser.add_argument(
        "-ymlIn", help="Read YAML representation from file (skip parsing)"
    )
    parser.add_argument("-verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "-obfuscate", action="store_true", help="Obfuscate function names"
    )
    parser.add_argument(
        "--renderer",
        choices=["auto", "python-graphviz", "system-dot", "networkx"],
        default="auto",
        help="Renderer preference",
    )
    args = parser.parse_args(argv)

    if not args.paths and not args.jsnIn and not args.ymlIn:
        parser.print_help()
        return 1

    tmpdir = None
    output = args.output
    files: List[str] = []

    if args.jsnIn:
        with open(args.jsnIn, "r", encoding="utf-8") as fh:
            call_graph = json.load(fh)
    elif args.ymlIn:
        if yaml is None:
            raise SystemExit("ERROR: PyYAML not available to read YAML input.")
        with open(args.ymlIn, "r", encoding="utf-8") as fh:
            call_graph = yaml.safe_load(fh)
    else:
        files = collect_files(args.paths, language=args.language)
        language = args.language or get_script_type(files[0], scripts_only=True)
        if not language:
            raise SystemExit(
                "ERROR: language could not be determined. Use -language <language>"
            )
        syntax = define_syntax(language)
        parse = parse_files(files, language, syntax)

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

        if args.writeFunctions:
            tmpdir = tempfile.mkdtemp(prefix="call_graph_funcs_")
            for func in parse.func_contents:
                for fpath in parse.func_contents[func]:
                    bfile = os.path.basename(fpath)
                    name, ext = os.path.splitext(bfile)
                    out = os.path.join(tmpdir, f"{name}__{func}{ext}")
                    say("Creating function source file", out)
                    with open(out, "w", encoding="utf-8") as oh:
                        oh.write(parse.func_contents[func][fpath])

        call_graph = build_call_graph(
            parse, files, ignore_re=args.ignore, language=language
        )

        if args.writeSubsetCode:
            subset_file = args.writeSubsetCode
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

        if args.jsnOut:
            with open(args.jsnOut, "w", encoding="utf-8") as oh:
                json.dump(call_graph, oh, indent=2)
            say(f"Wrote JSON to {args.jsnOut}")
        if args.ymlOut:
            if yaml is None:
                raise SystemExit(
                    "ERROR: PyYAML not available to write YAML output."
                )
            with open(args.ymlOut, "w", encoding="utf-8") as oh:
                yaml.safe_dump(call_graph, oh)
            say(f"Wrote YAML to {args.ymlOut}")

    if args.obfuscate:
        call_graph = obfuscate_call_graph(call_graph, ignore=["__MAIN__"])

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

    graph_builder = GraphBuilder(call_graph, cluster_files=False)
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

    graph_builder.generate_dot(dot_name, files, full_path=args.fullPath)

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

        def _try_python_graphviz():
            if "_graphviz" in globals() and _graphviz is not None:
                try:
                    say("Rendering via python 'graphviz' package to", image)
                    src = _graphviz.Source.from_file(dot_name)
                    outbase = os.path.splitext(output)[0]
                    src.format = fmt
                    rendered_path = src.render(filename=outbase, cleanup=False)
                    if isinstance(rendered_path, str) and os.path.exists(
                        rendered_path
                    ):
                        return rendered_path
                    alt = outbase + "." + fmt
                    if os.path.exists(alt):
                        return alt
                except Exception as e:
                    say("WARNING: python graphviz package failed to render:", e)
            return None

        def _try_system_dot():
            if which_command("dot"):
                try:
                    say("Rendering via system 'dot' to", image)
                    if fmt == "svg":
                        cmd = ["dot", "-Tsvg", dot_name, "-o", image]
                    elif fmt == "pdf":
                        cmd = ["dot", "-Tpdf", dot_name, "-o", image]
                    else:
                        cmd = ["dot", "-Tpng", dot_name, "-o", image]
                    subprocess.run(cmd, check=False)
                    if os.path.exists(image):
                        return image
                except Exception as e:
                    say("WARNING: system 'dot' failed to render:", e)
            return None

        def _try_networkx():
            try:
                ok = _render_with_networkx(graph_builder, image)
                if ok and os.path.exists(image):
                    say("Rendering via networkx/matplotlib to", image)
                    return image
            except Exception as e:
                say("WARNING: networkx/matplotlib render failed:", e)
            return None

        if renderer == "auto":
            path = _try_python_graphviz()
            if path:
                image = path
                rendered_ok = True
            else:
                path = _try_system_dot()
                if path:
                    image = path
                    rendered_ok = True
                else:
                    path = _try_networkx()
                    if path:
                        image = path
                        rendered_ok = True
        elif renderer == "python-graphviz":
            path = _try_python_graphviz()
            if path:
                image = path
                rendered_ok = True
            else:
                say(
                    "ERROR: python 'graphviz' package not available or failed to render as requested by --renderer"
                )
                return 1
        elif renderer == "system-dot":
            path = _try_system_dot()
            if path:
                image = path
                rendered_ok = True
            else:
                say(
                    "ERROR: system 'dot' not available or failed to render as requested by --renderer"
                )
                return 1
        elif renderer == "networkx":
            path = _try_networkx()
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
        if not args.noShow:
            viewer = which_command(
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

    # Verbose analysis
    if args.verbose and not (args.jsnIn or args.ymlIn):
        say(
            "\nVerbose global variable and external script analysis (best-effort):"
        )
        parse_result = parse_files(
            files,
            args.language or get_script_type(files[0]),
            define_syntax(args.language or get_script_type(files[0])),
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
                var_re = LANG_SYNTAX["variable"].get(
                    args.language or get_script_type(files[0]), None
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
