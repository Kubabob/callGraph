#!/usr/bin/env python3
from __future__ import annotations

"""
treesitter_wrapper.py

Conservative, opt-in Tree-sitter wrapper for callGraph.

- Provides a `TreesitterParser` class with a constructor accepting:
    - bundle_path: optional path to a combined Tree-sitter language bundle (.so/.dll/.dylib)
    - verbose: bool for extra logging

- Public method:
    - parse_files(files: List[str], language: str) -> ParseResult

Notes / design choices:
- This wrapper attempts to use `py-tree-sitter` if available. If the package is
  not importable, the constructor will raise ImportError so callers can fall
  back to regex parsing (the calling code in `callGraph.py` already handles that).
- For the initial POC we support C++ (language key 'cpp') only. If another
  language is requested the parser raises a ValueError.
- Loading a language:
    - If `bundle_path` is provided we call `Language(bundle_path, lang_name)`.
      This is the recommended flow when you supply a prebuilt combined bundle.
    - If `bundle_path` is None we make a best-effort attempt to import a
      per-language helper package (for example, `tree_sitter_cpp`) if present.
      Many environments (or system packaging) provide these; if not present
      the constructor will raise and the caller will fall back.
- The extraction is conservative:
    - We search for `function_definition` nodes to find functions.
    - We search for `call_expression` nodes to find calls.
    - Function names and called identifiers are extracted by looking for the
      first descendant node whose type contains "identifier".
    - Calls are attributed to the enclosing function by byte-range containment.
- The returned structure is compatible with `parsing.ParseResult` (same shape).

This module intentionally avoids any build-time assumptions and will not try to
compile grammars or write files. If you want to use Tree-sitter on a host that
doesn't have the language libs prebuilt, build a language bundle externally and
pass it via `--treesitter-bundle` when invoking the CLI.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

# Local import of the ParseResult dataclass used by the rest of the codebase.
# We import at runtime so this module can be imported only when needed.
try:
    import parsing  # type: ignore
except (
    Exception
):  # pragma: no cover - environment may not have parsing when linting
    parsing = None  # type: ignore

# Lazy import placeholders for py-tree-sitter bindings
_tree_sitter_available = False
Language: Any = None  # type: ignore
Parser: Any = None  # type: ignore

# Attempt several common module names/locations for the py-tree-sitter bindings.
# Some packaging/distributions expose the bindings under slightly different
# names. Be defensive: try a small list of plausible module names and extract
# `Language` and `Parser` if available.
_import_names = [
    "tree_sitter",
    "py_tree_sitter",
    "tree_sitter_binding",
    "tree_sitter_bindings",
]
for _mod in _import_names:
    try:
        mod = __import__(_mod)
    except Exception:
        continue
    # Try direct attributes first
    Lang = getattr(mod, "Language", None)
    Pars = getattr(mod, "Parser", None)
    # If not present, attempt to import with fromlist to handle some packaging quirks
    if Lang is None or Pars is None:
        try:
            tmp = __import__(_mod, fromlist=["Language", "Parser"])
            Lang = getattr(tmp, "Language", Lang)
            Pars = getattr(tmp, "Parser", Pars)
        except Exception:
            pass
    if Lang is not None and Pars is not None:
        Language = Lang
        Parser = Pars
        _tree_sitter_available = True
        break


class TreesitterParser:
    """
    Conservative wrapper around py-tree-sitter for optional parsing.

    Usage:
        ts = TreesitterParser(bundle_path="/path/to/libmylangs.so", verbose=True)
        parse_result = ts.parse_files(["a.cpp", "b.cpp"], "cpp")
    """

    def __init__(
        self, bundle_path: Optional[str] = None, verbose: bool = False
    ):
        if parsing is None:
            raise ImportError(
                "Internal error: required local module 'parsing' could not be imported."
            )
        self.bundle_path = bundle_path
        self.verbose = verbose

        if not _tree_sitter_available:
            raise ImportError(
                "py-tree-sitter (tree_sitter) not available. Install it or run without --parser treesitter."
            )

        # Cache of loaded Language objects keyed by language name
        self._languages: Dict[str, Any] = {}
        # Cache of Parsers per language
        self._parsers: Dict[str, Any] = {}

    def _log(self, *parts) -> None:
        if self.verbose:
            print("TreesitterParser:", *parts)

    def _load_language(self, lang_name: str):
        """
        Attempt to load a Language for `lang_name`.

        - If a bundle_path was provided, attempt `Language(bundle_path, lang_name)`.
        - Otherwise, attempt to import a convenience language package such as
          `tree_sitter_cpp` and extract a Language from it (best-effort).
        - Raises RuntimeError on failure with a helpful message.
        """
        if lang_name in self._languages:
            return self._languages[lang_name]

        # 1) Attempt to load from provided bundle_path
        if self.bundle_path:
            try:
                self._log(
                    "Loading language",
                    lang_name,
                    "from bundle",
                    self.bundle_path,
                )
                lang = Language(self.bundle_path, lang_name)
                self._languages[lang_name] = lang
                return lang
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load language '{lang_name}' from bundle '{self.bundle_path}': {e}"
                ) from e

        # 2) Attempt to import a per-language shim package (best-effort).
        # Example package names: tree_sitter_cpp, tree_sitter_c, tree_sitter_rust
        shim_names = [f"tree_sitter_{lang_name}", f"tree-sitter-{lang_name}"]
        for shim in shim_names:
            try:
                mod = __import__(shim)
                # Common conventions: module exposes `Language` or `get_language()` or `language()` or `LANG`
                # 1) Some language shim packages provide a `get_language()` helper.
                if hasattr(mod, "get_language"):
                    try:
                        lang = mod.get_language()
                        self._languages[lang_name] = lang
                        self._log(
                            "Loaded language via get_language() from", shim
                        )
                        return lang
                    except Exception:
                        pass
                # 1b) Some language shim packages expose a `language()` function (e.g. tree_sitter_cpp.language())
                if hasattr(mod, "language"):
                    try:
                        lang = mod.language()
                        # Accept the returned object as the language representation.
                        # It may be a PyCapsule or a Language-like object depending on packaging.
                        self._languages[lang_name] = lang
                        self._log("Loaded language via language() from", shim)
                        return lang
                    except Exception:
                        pass
                # 2) Some shims expose a Language instance or factory
                if hasattr(mod, "Language"):
                    candidate = getattr(mod, "Language")
                    if isinstance(candidate, type):
                        # If it's a class, we can't safely instantiate it without args.
                        # Skip and fall back to next option.
                        pass
                    else:
                        # If it's an already-built Language object, use it.
                        self._languages[lang_name] = candidate
                        self._log("Loaded Language instance from", shim)
                        return candidate
                # 3) Some shims expose a top-level `LANG` or similar
                for attr in ("LANG", "LANGUAGE", "CPP", "C"):
                    if hasattr(mod, attr):
                        candidate = getattr(mod, attr)
                        self._languages[lang_name] = candidate
                        self._log("Loaded language from", shim, "attr", attr)
                        return candidate
            except ImportError:
                continue
            except Exception:
                # Don't fail fast on heuristic attempts; continue trying other options.
                continue

        # 3) Give up with a helpful error
        raise RuntimeError(
            "Could not locate a Tree-sitter language for '{}'.\n"
            "Options:\n"
            "  - Provide a combined language bundle via --treesitter-bundle <path>.\n"
            "  - Install a per-language Python package (e.g. a prebuilt wheel) that "
            "exposes a Language object to import.\n"
            "  - Run without --parser treesitter to use the conservative regex fallback.".format(
                lang_name
            )
        )

    def _get_parser_for(self, lang_name: str) -> Any:
        if lang_name in self._parsers:
            return self._parsers[lang_name]

        lang = self._load_language(lang_name)

        # Different py-tree-sitter builds expose different Parser APIs. Try several
        # strategies to obtain a Parser instance configured for `lang`.
        p = None

        # Strategy 1: default constructor then call set_language (common API)
        try:
            p = Parser()  # type: ignore
            if hasattr(p, "set_language"):
                try:
                    p.set_language(lang)  # type: ignore
                except Exception:
                    # failed to set language on this parser instance; discard
                    p = None
            else:
                # Parser instance doesn't support set_language; discard
                p = None
        except Exception:
            p = None

        # Strategy 2: constructor that accepts language directly (some builds)
        if p is None:
            try:
                p = Parser(lang)  # type: ignore
            except Exception:
                p = None

        # Strategy 3: coerce the loaded language (e.g. Raw capsule -> Language) and retry
        if p is None:
            try:
                coerced = Language(lang)  # type: ignore
                try:
                    p = Parser()  # type: ignore
                    if hasattr(p, "set_language"):
                        p.set_language(coerced)  # type: ignore
                    else:
                        # attempt constructor-with-arg fallback
                        p = Parser(coerced)  # type: ignore
                except Exception:
                    # final attempt: try constructor with coerced language
                    p = Parser(coerced)  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    f"Failed to instantiate Parser for {lang_name}: {e}"
                ) from e

        # If we still don't have a parser, surface a helpful error.
        if p is None:
            raise RuntimeError(
                f"Could not create a configured Parser for language '{lang_name}'"
            )

        self._parsers[lang_name] = p
        return p

    @staticmethod
    def _find_first_identifier(node) -> Optional[Tuple[int, int]]:
        """
        Walk descendants of `node` to find the first identifier-like node.
        Returns a tuple (start_byte, end_byte) referencing the source bytes slice,
        or None if no identifier-like node is found.

        We consider a node to be identifier-like if its type contains 'identifier'.
        """
        # Breadth-first search to prefer nearer identifiers (e.g. function names)
        queue = [node]
        while queue:
            n = queue.pop(0)
            try:
                t = getattr(n, "type", "")
            except Exception:
                t = ""
            if "identifier" in t.lower():
                return (n.start_byte, n.end_byte)
            # push children
            try:
                queue.extend(list(n.children))
            except Exception:
                pass
        return None

    def parse_files(self, files: List[str], language: str) -> Any:
        """
        Parse the given files and return a ParseResult compatible object.

        Currently only supports C++ (language == 'cpp'). The function is
        conservative: when it cannot extract an identifier for a function or
        call site it will use synthetic names to avoid crashing downstream code.
        """
        if language not in ("cpp", "c++", "c"):
            raise ValueError(
                f"TreesitterParser currently supports C/C++ via language key 'cpp' (requested '{language}')."
            )

        # canonicalize to 'cpp' for language loader
        lang_key = (
            "cpp" if language.startswith("c") or "cpp" in language else language
        )

        parser = self._get_parser_for(lang_key)

        main_key = "__MAIN__"
        func_contents: Dict[str, Dict[str, str]] = defaultdict(dict)
        func_definition: Dict[str, Dict[str, int]] = defaultdict(dict)
        func_call_nested = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        shebang: Optional[str] = None

        for fpath in files:
            try:
                with open(fpath, "rb") as fh:
                    src_bytes = fh.read()
                try:
                    text = src_bytes.decode("utf-8")
                except Exception:
                    text = src_bytes.decode("utf-8", errors="ignore")
            except Exception as e:
                self._log("Could not read", fpath, ":", e)
                src_bytes = b""
                text = ""

            if text:
                first_line = text.splitlines()[0] if text.splitlines() else ""
                if first_line.startswith("#!"):
                    shebang = first_line.strip()

            if not src_bytes:
                continue

            tree = parser.parse(src_bytes)

            root = tree.root_node

            # Collect function nodes: nodes of type 'function_definition'
            # We'll gather a list of tuples: (name, node, start_byte, end_byte, lineno)
            functions_meta: List[Tuple[str, object, int, int, int]] = []

            # traverse tree to collect function_definition nodes
            to_visit = [root]
            while to_visit:
                node = to_visit.pop()
                try:
                    ntype = node.type
                except Exception:
                    ntype = ""
                if ntype == "function_definition":
                    # Attempt to find a name inside this node
                    ident_span = self._find_first_identifier(node)
                    if ident_span:
                        start_b, end_b = ident_span
                        name = src_bytes[start_b:end_b].decode(
                            "utf-8", errors="ignore"
                        )
                    else:
                        # Fallback synthetic name with start row
                        name = f"<anon_fn_{node.start_point[0] + 1}>"
                    start_byte = node.start_byte
                    end_byte = node.end_byte
                    lineno = node.start_point[0] + 1
                    functions_meta.append(
                        (name, node, start_byte, end_byte, lineno)
                    )
                # We'll still recurse into children
                try:
                    to_visit.extend(list(node.children))
                except Exception:
                    pass

            # Sort functions by start position (ascending)
            functions_meta.sort(key=lambda t: t[2])

            # Build mappings for quick containment checks
            # For each function we will extract source slice and record definition info.
            for name, node, sbyte, ebyte, lineno in functions_meta:
                try:
                    contents = src_bytes[sbyte:ebyte].decode(
                        "utf-8", errors="ignore"
                    )
                except Exception:
                    contents = ""
                func_contents[name][fpath] = contents
                func_definition[name][fpath] = lineno

            # At this point, any call expression will be attributed to the function
            # whose byte-range contains the call start. If no function contains it,
            # the caller is __MAIN__.
            # Find call_expression nodes
            call_nodes = []
            to_visit = [root]
            while to_visit:
                node = to_visit.pop()
                try:
                    ntype = node.type
                except Exception:
                    ntype = ""
                if ntype == "call_expression":
                    call_nodes.append(node)
                try:
                    to_visit.extend(list(node.children))
                except Exception:
                    pass

            # Helper: find containing function name for a byte offset
            def find_enclosing_function_name(offset: int) -> str:
                for name, _, sbyte, ebyte, _ in functions_meta:
                    if sbyte <= offset < ebyte:
                        return name
                return main_key

            # For each call node, extract an identifier and increment counters
            for cn in call_nodes:
                # attempt to find identifier in the function position or descendents
                # Many C++ call expressions have a child representing the function
                # target; we'll search descendants for identifier-like nodes.
                ident_span = self._find_first_identifier(cn)
                if ident_span:
                    start_b, end_b = ident_span
                    called_name = src_bytes[start_b:end_b].decode(
                        "utf-8", errors="ignore"
                    )
                else:
                    # fallback: try to extract a short text snippet for diagnostics
                    try:
                        snippet = src_bytes[cn.start_byte : cn.end_byte].decode(
                            "utf-8", errors="ignore"
                        )
                        # crude heuristic to pick an alphanumeric token
                        import re as _re

                        m = _re.search(r"[A-Za-z_]\w*", snippet)
                        called_name = m.group(0) if m else "<call>"
                    except Exception:
                        called_name = "<call>"

                caller = find_enclosing_function_name(cn.start_byte)
                func_call_nested[caller][fpath][called_name] += 1

            # If no functions were discovered we still want to record __MAIN__ contents
            if not functions_meta:
                func_contents[main_key].setdefault(fpath, "")
                func_contents[main_key][fpath] += text

        # Normalize into plain dicts to match parsing.ParseResult semantics
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

        # Build and return the ParseResult dataclass from the parsing module if available.
        if parsing is not None and hasattr(parsing, "ParseResult"):
            return parsing.ParseResult(
                shebang=shebang,
                func_contents=func_contents_out,
                func_definition=func_definition_out,
                func_call=func_call_out,
            )

        # Fallback: create a minimal compatible object when parsing.ParseResult is unavailable.
        # This ensures callers (the rest of the tool) can rely on the same attribute names.
        class _LocalParseResult:
            def __init__(
                self, shebang, func_contents, func_definition, func_call
            ):
                self.shebang = shebang
                self.func_contents = func_contents
                self.func_definition = func_definition
                self.func_call = func_call

        return _LocalParseResult(
            shebang=shebang,
            func_contents=func_contents_out,
            func_definition=func_definition_out,
            func_call=func_call_out,
        )
