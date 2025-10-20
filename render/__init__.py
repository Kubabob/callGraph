#!/usr/bin/env python3
from __future__ import annotations

"""
render package: rendering helpers for callGraph.

Exports:
- try_python_graphviz(dot_name, output, fmt) -> Optional[str]
- try_system_dot(dot_name, image, fmt) -> Optional[str]
- try_networkx(graph_builder, image) -> Optional[str]
- which_command(*names) -> Optional[str]

These functions try different backends to convert a DOT file into an image:
1) python-graphviz (graphviz Python package)
2) system `dot` executable
3) networkx + matplotlib fallback

All functions are best-effort: they return the path to the generated image on
success, or None on failure.
"""

import os
import subprocess
import sys
from typing import Optional

try:
    # Import Graphviz python bindings if available
    import graphviz as _graphviz  # type: ignore
    from graphviz import Source  # type: ignore
except Exception:
    _graphviz = None
    Source = None


def which_command(*names: str) -> Optional[str]:
    """
    Return the first available command from `names` using shutil.which, or None.
    """
    import shutil

    for n in names:
        p = shutil.which(n)
        if p:
            return n
    return None


def try_python_graphviz(dot_name: str, output: str, fmt: str) -> Optional[str]:
    """
    Attempt to render the DOT file using the python 'graphviz' package.

    - dot_name: path to the .dot file
    - output: desired final output path (used to determine base name)
    - fmt: desired output format ('png', 'svg', 'pdf', ...)

    Returns path to generated image on success, else None.
    """
    if (
        "_graphviz" in globals()
        and _graphviz is not None
        and Source is not None
    ):
        try:
            outbase = os.path.splitext(output)[0]
            src = Source.from_file(dot_name)
            src.format = fmt
            # render returns the path to the rendered file when successful
            rendered_path = src.render(filename=outbase, cleanup=False)
            if isinstance(rendered_path, str) and os.path.exists(rendered_path):
                return rendered_path
            alt = outbase + "." + fmt
            if os.path.exists(alt):
                return alt
        except Exception:
            return None
    return None


def try_system_dot(dot_name: str, image: str, fmt: str) -> Optional[str]:
    """
    Attempt to render the DOT file using the system 'dot' command.

    - dot_name: path to the .dot file
    - image: desired output image path
    - fmt: output format

    Returns image path on success, else None.
    """
    if which_command("dot"):
        try:
            if fmt == "svg":
                cmd = ["dot", "-Tsvg", dot_name, "-o", image]
            elif fmt == "pdf":
                cmd = ["dot", "-Tpdf", dot_name, "-o", image]
            else:
                cmd = ["dot", "-Tpng", dot_name, "-o", image]
            subprocess.run(cmd, check=False)
            if os.path.exists(image):
                return image
        except Exception:
            return None
    return None


def try_networkx(graph_builder, image: str) -> Optional[str]:
    """
    Pure-Python fallback renderer using networkx + matplotlib.

    - graph_builder: GraphBuilder instance with `.node` and `.edge` attributes
    - image: desired output image path (extension determines format)

    Returns image path on success, else None.
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except Exception:
        return None

    try:
        G = nx.DiGraph()
        labels = {}
        node_colors = []
        for n in sorted(graph_builder.node.keys()):
            G.add_node(n)
            # Node label: basename(file) + newline + function
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

        ext = os.path.splitext(image)[1].lower()
        if ext == ".svg":
            plt.savefig(image, format="svg", bbox_inches="tight")
        elif ext == ".pdf":
            plt.savefig(image, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(image, format="png", bbox_inches="tight")
        plt.close()
        return image
    except Exception:
        return None
