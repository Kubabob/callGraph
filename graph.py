#!/usr/bin/env python3
from __future__ import annotations

"""
graph.py

Graph construction and DOT generation utilities extracted from the monolithic
callGraph script.

Exports:
- GraphBuilder: incremental call-graph traversal and DOT generation
- build_call_graph(parse, files, ignore_re=None, language=None): produce
  a normalized call graph dict from a ParseResult
- obfuscate_call_graph(call_graph, ignore=None, seed=None): return a
  renamed/obfuscated copy of a call graph
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Optional


class GraphBuilder:
    """
    Build a graph of function nodes and edges from a call-graph mapping and
    provide DOT generation helpers.

    The builder records nodes visited via `plot(...)` and recursively visits
    parents/children depending on the requested direction. After plotting the
    desired starting nodes, call `generate_dot(dot_name, files, full_path=False)`
    to write a DOT file.
    """

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
        """
        Add `from_file_sub` as a node and recursively include parents/children
        based on the `direction` string which may contain 'up' and/or 'down'.
        If direction is omitted, defaults to 'up down' (both).
        """
        direction = direction or "up down"
        self.node[from_file_sub] = self.node.get(from_file_sub, 0) + 1
        if direction == "up down":
            # mark as initial node for styling (e.g. filled)
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
        """
        Write a DOT representation of the currently recorded graph to `dot_name`.

        Tries to use the `graphviz` Python package if available to build the DOT
        source; otherwise falls back to a plaintext DOT writer.
        """
        nodes = sorted(self.node.keys())
        id_map = {n: self._sanitize_node_id(n) for n in nodes}

        # Attempt to use python-graphviz if installed to generate consistent DOT
        try:
            from graphviz import Digraph  # type: ignore

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
            print("Generating:", dot_name)
            return
        except Exception:
            # Fall through to text-based writer
            pass

        # Fallback DOT writer (plain text)
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
        print("Generating:", dot_name)


def obfuscate_call_graph(
    call_graph: Dict[str, Dict],
    ignore: Optional[List[str]] = None,
    seed: Optional[int] = None,
):
    """
    Return a copy of `call_graph` in which function names are replaced by
    pseudo-random jargon terms. `ignore` may list names (e.g. '__MAIN__') to
    preserve. Optionally provide a `seed` for deterministic output.
    """
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


def build_call_graph(
    parse,
    files: List[str],
    ignore_re: Optional[str] = None,
    language: Optional[str] = None,
):
    """
    Construct a call graph from a ParseResult-like object. The return value is a
    dict mapping 'file:func' -> {'calls': {to: count}, 'called_by': {from: count}}.

    Behavior mirrors the original monolithic implementation:
    - Uses explicit parse.func_call entries
    - Performs conservative content-based and range-based rescans for non-Python
      languages to increase recall (these rescans are conservative and may be
      disabled later).
    """
    call_graph = defaultdict(lambda: {"calls": {}, "called_by": {}})

    # explicit parse entries
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
                # prefer same-file definition when possible
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
                        # ambiguous reference; skip
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
