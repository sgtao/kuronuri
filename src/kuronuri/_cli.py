"""Command-line interface for kuronuri."""

import codecs
import re
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer

from kuronuri._masker import (
    EN_MODEL,
    JA_MODEL,
    MaskStrategy,
    NERModel,
    mask,
    mask_with_block,
    mask_with_fixed,
    mask_with_label,
)

app = typer.Typer(help="Mask PII in text files.")

_NEWLINE_RE = re.compile(r"\r\n|\r|\n")

# Ordered from longest to shortest BOM so prefix matching is unambiguous.
_BOM_ENCODINGS: list[tuple[bytes, str]] = [
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
    (codecs.BOM_UTF8, "utf-8-sig"),
]

_BUILTIN_MODELS: dict[str, NERModel] = {"en": EN_MODEL, "ja": JA_MODEL}


def _detect_encoding_and_bom(raw: bytes) -> tuple[str, bytes]:
    """Return ``(encoding, bom_bytes)`` by inspecting the raw bytes.

    *bom_bytes* is the raw BOM prefix that should be re-prepended on write,
    or ``b""`` if no BOM was found.  For ``utf-8-sig`` the Python codec strips
    the BOM automatically, so ``bom_bytes`` is left empty for that encoding.
    """
    for bom, enc in _BOM_ENCODINGS:
        if raw.startswith(bom):
            # utf-8-sig: Python's codec handles the BOM transparently.
            return enc, (b"" if enc == "utf-8-sig" else bom)

    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return sys.getdefaultencoding(), b""
    else:
        return "utf-8", b""


def _detect_newline(text: str) -> str:
    """Return the first newline sequence found in *text*, defaulting to LF."""
    m = _NEWLINE_RE.search(text)
    return m.group() if m else "\n"


def _normalize_newlines(text: str, newline: str) -> str:
    """Replace all newline sequences in *text* with *newline*."""
    return _NEWLINE_RE.sub(newline, text)


def _resolve_strategy(strategy_name: str, char: str, length: int) -> MaskStrategy:
    match strategy_name:
        case "label":
            return mask_with_label
        case "block":
            return mask_with_block
        case "fixed":
            return mask_with_fixed(char=char, length=length)
        case _:
            msg = f"Unknown strategy: {strategy_name!r}"
            raise typer.BadParameter(msg)


def _resolve_model(lang: str | None, model_name: str | None) -> NERModel:
    """Return the NERModel to use, applying CLI precedence rules."""
    if lang is not None and model_name is not None:
        msg = "Specify either --lang or --model, not both."
        raise typer.BadParameter(msg)
    if model_name is not None:
        return NERModel(model_name=model_name, default_mask_tags=frozenset())
    if lang is not None:
        if lang not in _BUILTIN_MODELS:
            available = ", ".join(sorted(_BUILTIN_MODELS))
            msg = f"Unknown lang {lang!r}. Available: {available}"
            raise typer.BadParameter(msg)
        return _BUILTIN_MODELS[lang]
    return EN_MODEL


def _version_callback(value: bool) -> None:  # noqa: FBT001
    if value:
        typer.echo(f"kuronuri v{version('kuronuri')}")
        raise typer.Exit


def _read_text(input_path: Path) -> tuple[str, str, bytes]:
    raw = input_path.read_bytes()
    encoding, bom_bytes = _detect_encoding_and_bom(raw)

    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        msg = f"Error: cannot decode {input_path} as {encoding}: {exc}"
        typer.echo(msg, err=True)
        raise typer.Exit(code=1) from exc

    if bom_bytes:
        bom_str = bom_bytes.decode(encoding, errors="ignore")
        text = text.removeprefix(bom_str)

    return text, encoding, bom_bytes


def _write_text(
    masked: str, encoding: str, bom_bytes: bytes, output_file: Path | None
) -> None:
    result_bytes = bom_bytes + masked.encode(encoding)

    if output_file is None:
        sys.stdout.buffer.write(result_bytes)
    else:
        output_file.write_bytes(result_bytes)


def _process_file(
    input_path: Path,
    output_file: Path | None,
    ner_model: NERModel,
    tags: frozenset[str] | None,
    mask_strategy: MaskStrategy,
) -> None:
    text, encoding, bom_bytes = _read_text(input_path)
    newline = _detect_newline(text)
    masked = _normalize_newlines(
        mask(text, model=ner_model, mask_tags=tags, strategy=mask_strategy), newline
    )
    _write_text(masked, encoding, bom_bytes, output_file)


@app.command()
def main(
    input_: Annotated[
        str,
        typer.Argument(
            help="Path to a text file, or a literal string to mask inline.",
            metavar="INPUT",
        ),
    ],
    output_file: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path. Ignored for inline text. Defaults to stdout.",
        ),
    ] = None,
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help=(
                "Masking strategy: 'block' (███, default),"
                " 'label' (<Person>), or 'fixed'."
            ),
        ),
    ] = "block",
    fixed_char: Annotated[
        str,
        typer.Option("--fixed-char", help="Character used by the 'fixed' strategy."),
    ] = "*",
    fixed_length: Annotated[
        int, typer.Option("--fixed-length", help="Length used by the 'fixed' strategy.")
    ] = 3,
    mask_tags: Annotated[
        list[str] | None,
        typer.Option(
            "--tag", "-t", help="Entity tag to mask. Can be specified multiple times."
        ),
    ] = None,
    lang: Annotated[
        str | None,
        typer.Option(
            "--lang",
            help=(
                "Built-in language shorthand: 'en' (default) or 'ja'. "
                "Mutually exclusive with --model."
            ),
        ),
    ] = None,
    model_name: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help=(
                "Hugging Face model identifier for a custom model. "
                "Mutually exclusive with --lang."
            ),
        ),
    ] = None,
    version_: Annotated[  # noqa: ARG001
        bool | None,
        typer.Option(
            "--version",
            "-v",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    """Mask PII in a text file or an inline string."""
    ner_model = _resolve_model(lang, model_name)
    tags = frozenset(mask_tags) if mask_tags else None
    mask_strategy = _resolve_strategy(strategy, fixed_char, fixed_length)

    input_path = Path(input_)
    if input_path.exists():
        _process_file(input_path, output_file, ner_model, tags, mask_strategy)
    else:
        typer.echo(
            mask(input_, model=ner_model, mask_tags=tags, strategy=mask_strategy)
        )
