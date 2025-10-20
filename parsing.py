#!/usr/bin/env python3
from __future__ import annotations

"""
parsing.py

Modular parsing utilities extracted from the monolithic callGraph script.
Provides:

- LANG_SYNTAX: conservative regex heuristics for fallback parsing
- ParseResult dataclass: normalized return structure for parsers
- define_syntax(language) -> compiled regex dict
- parse_files(files, language, language_syntax) -> ParseResult
    (uses AST for Python, conservative line-oriented regex fallback for other languages)
- get_script_type(file, scripts_only=False, forced_language=None) -> str
- collect_files(paths, language=None) -> List[str]

This module intentionally implements the same *behavior* as the original
monolithic parser to maintain backward compatibility with the rest of the
codebase. It is conservative about regex matches and keeps AST-based parsing
for Python.
"""

import ast
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

# Conservative heuristic regexes used as a fallback for non-Python languages.
# These mirror the original monolithic definitions and are intentionally
# permissive in some areas and conservative in others.
LANG_SYNTAX = {
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
        # Block terminators for block-oriented languages; Python is dedent-based
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
        # Very basic call-site heuristic: identifier followed by '('
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
        "py": r"([A-Za-z_]\w+)",
        "js": r"([A-Za-z_]\w+)",
        "ts": r"([A-Za-z_]\w+)",
        "rs": r"([A-Za-z_]\w+)",
        "rb": r"([A-Za-z_]\w+)",
        "php": r"([A-Za-z_]\w+)",
    },
}


# Normalized parse result returned by parse_files
@dataclass
class ParseResult:
    shebang: Optional[str]
    func_contents: Dict[str, Dict[str, str]]  # func -> file -> contents
    func_definition: Dict[str, Dict[str, int]]  # func -> file -> first line
    func_call: Dict[
        str, Dict[str, Dict[str, int]]
    ]  # caller_func -> file -> called_name -> count


def define_syntax(language: str):
    """
    Given a language key (as returned by get_script_type), return a dict with
    compiled regex patterns for the keys:
      - functionDefinition
      - functionEnd
      - functionCall
      - comment
      - variable

    If a pattern is not defined for the requested language the value will be None.
    """
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

    - For Python files ('py') this uses the `ast` module (accurate).
    - For other languages it falls back to a conservative, line-oriented regex
      heuristic (the patterns are exposed in LANG_SYNTAX).
    - Returns a ParseResult with normalized plain dicts (not defaultdicts).
    """
    main = "__MAIN__"
    func_contents = defaultdict(dict)  # func -> file -> contents
    func_definition = defaultdict(dict)  # func -> file -> first-line
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
                        name = getattr(node, "name", None)
                        lineno = getattr(node, "lineno", None)
                        if name and lineno:
                            func_definition[name][file] = lineno
                            func_contents[name].setdefault(file, "")
                            func_contents[name][file] += source_for(node) + "\n"
                        self.current_stack.append(name or "<anon>")
                        self.generic_visit(node)
                        self.current_stack.pop()

                    def visit_AsyncFunctionDef(self, node: ast.AST):
                        return self.visit_FunctionDef(node)

                    def visit_ClassDef(self, node: ast.AST):
                        # Traverse class body to find nested calls but don't treat class itself
                        # as a function definition.
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
                        # Record module-level statements under __MAIN__, but still visit
                        # every statement so nested FunctionDef handlers run.
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
                            # always visit so FunctionDef/ClassDef are processed
                            self.visit(stmt)

                visitor = PyVisitor()
                visitor.visit(tree)
                # AST parse handled the file; skip the regex fallback
                continue

        # Fallback parsing (line-oriented heuristics)
        norm_text = text.replace("\\\n", "")
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

            # perl POD / __END__ best-effort handling
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

            # append blank/comment-only lines to current function contents
            if re.match(r"^\s*(#.*)?$", line):
                func_contents[func_stack[-1]].setdefault(file, "")
                func_contents[func_stack[-1]][file] += line + "\n"
                continue

            # strip single-line comments (best-effort)
            if comment_re:
                try:
                    line = re.sub(rf"{comment_re.pattern}.*", "", line)
                except re.error:
                    # ignore regex errors and proceed with the original line
                    pass

            # detect function definition
            if fd_re:
                m = fd_re.match(line)
                if m:
                    groups = [g for g in m.groups() if g is not None]
                    leading_spaces = groups[0] if groups else ""
                    func_name = ""
                    # prefer the last group that looks like an identifier
                    for g in reversed(groups):
                        try:
                            g_str = str(g)
                        except Exception:
                            continue
                        if re.match(r"^[A-Za-z_]\w*$", g_str):
                            func_name = g_str
                            break
                    # fallback: look at lastindex
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
                            # function ended on same line (C-like style)
                            pass
                        else:
                            func_stack.append(func_name)
                            space_stack.append(leading_spaces)
                    continue

            # detect explicit function end for block languages
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

            # detect call sites (simple heuristic)
            if fc_re:
                for mcall in fc_re.finditer(line):
                    try:
                        called = mcall.group(1)
                    except Exception:
                        called = None
                    if language in ("tcl", "pl") and called:
                        called = called.split("::")[-1]
                    if called:
                        func_call_nested[func_stack[-1]][file][called] += 1

    # Normalize nested defaultdicts into plain dicts
    func_contents_out: Dict[str, Dict[str, str]] = {
        fn: dict(fmap) for fn, fmap in func_contents.items()
    }
    func_definition_out: Dict[str, Dict[str, int]] = {
        fn: dict(fmap) for fn, fmap in func_definition.items()
    }
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


def get_script_type(
    file: str, scripts_only: bool = False, forced_language: Optional[str] = None
) -> str:
    """
    Determine a conservative language identifier for a file path.

    - Honors `forced_language` when provided.
    - Looks at file extension; recognizes common extensions including
      jsx/tsx.
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
    # shebang sniffing
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
    # final fallback
    return (
        "" if scripts_only else (file.rsplit(".", 1)[-1] if "." in file else "")
    )


def collect_files(
    paths: List[str], language: Optional[str] = None
) -> List[str]:
    """
    Given a list of file paths and/or directories, collect matching files.

    When a directory is provided, `language` must be specified and files inside
    the directory are filtered by language (via get_script_type).
    """
    found: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            if not language:
                raise SystemExit(
                    "ERROR: Must specify --language when scanning a directory."
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
