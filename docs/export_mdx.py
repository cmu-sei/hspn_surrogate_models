#!/usr/bin/env python3
"""Export Python package docs as MDX.

Usage:
    export_mdx.py mypkg --out docs/pages/api/mypkg.mdx
"""

import argparse
import logging
from pathlib import Path
from typing import cast

from griffe import (
    Alias,
    Class,
    DocstringSection,
    DocstringSectionParameters,
    DocstringSectionRaises,
    DocstringSectionReturns,
    Expr,
    Function,
    Object,
    load,
    parse_google,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(funcName)s] %(message)s",
)

logger = logging.getLogger(__name__)


def md_heading(level: int, text: str) -> str:
    return f"{'#' * level} {text}\n"


def render_docstring_sections(sections: list[DocstringSection]) -> str:
    out = []
    for sec in sections:
        kind = sec.kind
        if kind == "text":
            out.append(sec.value)
        elif kind == "parameters":
            sec = cast("DocstringSectionParameters", sec)
            out.append("\n**Parameters**\n")
            out.extend(f"  - **{p.name}**: {p.description}" for p in sec.value)
        elif kind == "returns":
            sec = cast("DocstringSectionReturns", sec)
            out.append("\n**Returns**\n")
            for r in sec.value:
                entry = "  "
                if r.name:
                    entry += f"**{r.name}**: "
                entry += r.description
                out.append(entry)
        elif kind == "raises":
            sec = cast("DocstringSectionRaises", sec)
            out.append("\n**Raises**\n")
            out.extend(f"  **{e.annotation}**: {e.description}" for e in sec.value)
        else:
            print("Unknown section", sec.kind)
            print(sec)
    return "\n".join(out).strip()


def render_function(fn: Function, level: int) -> str:
    params = []
    for p in fn.parameters:
        param_str = p.name
        if ann := p.annotation:
            if isinstance(ann, Expr):
                ann = ann.modernize().canonical_name
            param_str += ": " + ann
        params.append(param_str)
    param_str = ", ".join(params)
    sig = fn.name + "(" + param_str + ")"
    if fn.returns:
        sig += f" -> {fn.returns}"
    sections = parse_google(fn.docstring) if fn.docstring else []
    parts = [md_heading(level, fn.name), "```python", sig, "```", "", render_docstring_sections(sections), ""]
    if fn.filepath:
        parts += [
            "<details>\n<summary><i>Source Code</i></summary>\n",
            "```python",
            fn.source,
            "```",
            "</details>\n",
        ]
    return "\n".join(p for p in parts if p)


def render_class(cls: Class, level: int = 2) -> str:
    parts = [
        md_heading(level, cls.name),
        "",
    ]
    if cls.docstring:
        parts.extend((render_docstring_sections(parse_google(cls.docstring)), ""))
    for m in cls.members.values():
        if not m.is_function:
            continue
        if m.name.startswith("_"):
            logger.debug("Excluding method %s", m)
            continue
        assert isinstance(m, Function)
        parts.append(render_function(m, level + 1))
    return "\n".join(parts)


def render_module(root_module: Object | Alias, level: int = 1) -> str:
    def rec(mod: Object | Alias, parts):
        if mod.is_namespace_package:
            logger.info("Excluding heading for namespace package %s", mod.name)
        elif mod.is_namespace_subpackage:
            logger.info("Excluding heading for namespace subpackage %s", mod.name)
        else:
            logger.info("Rendering module %s (%d)", mod.name, level)
            parts.extend([md_heading(level, mod.name), ""])
        if mod.docstring:
            parts.extend((render_docstring_sections(parse_google(mod.docstring)), ""))
        for member in mod.members.values():
            if member.is_alias:
                continue
            if member.is_module:
                if member.name.startswith("test"):
                    logger.debug("Excluding module %s", member)
                    continue
                assert isinstance(member, Object)
                rec(member, parts)
            if member.is_class:
                assert isinstance(member, Class)
                parts.append(render_class(member, level + 1))
            elif member.is_function:
                if member.name.startswith("_"):
                    logger.debug("Excluding function %s", member)
                    continue
                assert isinstance(member, Function)
                parts.append(render_function(member, level + 1))

    results = []
    rec(root_module, results)
    return "\n".join(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    mod: Object | Alias = load(args.package)
    mdx = "\n".join([
        "---",
        f"title: {mod.name} API reference",
        f"description: {mod.docstring.value.splitlines()[0] if mod.docstring else ''}",
        "showOutline: 5",
        "---",
        "",
        render_module(mod, 2),
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(mdx, encoding="utf-8")
    logger.info(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
