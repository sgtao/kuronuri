"""MCP server for kuronuri — PII masking tools via the Model Context Protocol."""

from mcp.server.fastmcp import FastMCP

from kuronuri._masker import (
    EN_MODEL,
    JA_MODEL,
    mask,
    mask_with_block,
    mask_with_fixed,
    mask_with_label,
)

mcp = FastMCP("kuronuri")


@mcp.tool()
def mask_text(
    text: str,
    lang: str = "en",
    strategy: str = "block",
    mask_tags: list[str] | None = None,
    fixed_char: str = "█",
    fixed_length: int = 3,
) -> str:
    """Mask PII in the given text using kuronuri and return the result.

    Args:
        text: Input text to mask.
        lang: Language code ("en" or "ja").
        strategy: Masking strategy ("block" / "label" / "fixed").
        mask_tags: List of NER tags to mask. None uses the model default.
        fixed_char: Replacement character for the fixed strategy.
        fixed_length: Number of replacement characters for the fixed strategy.

    Returns:
        Masked text.
    """
    model = JA_MODEL if lang == "ja" else EN_MODEL

    match strategy:
        case "label":
            mask_strategy = mask_with_label
        case "fixed":
            mask_strategy = mask_with_fixed(char=fixed_char, length=fixed_length)
        case _:
            mask_strategy = mask_with_block

    tags: frozenset[str] | None = frozenset(mask_tags) if mask_tags else None
    return mask(text, model=model, mask_tags=tags, strategy=mask_strategy)


@mcp.tool()
def list_ner_tags(lang: str = "en") -> str:
    """Return a Markdown table of NER tags and default mask targets for the given lang.

    Args:
        lang: Language code ("en" or "ja").

    Returns:
        Markdown table of tag information.
    """
    model = JA_MODEL if lang == "ja" else EN_MODEL

    lines = [
        f"## {lang.upper()} Model: `{model.model_name}`\n",
        "| Tag | Label | Default Mask |",
        "|---|---|---|",
    ]
    for tag, label in model.tag_labels.items():
        default = "✅" if tag in model.default_mask_tags else "—"
        lines.append(f"| `{tag}` | {label} | {default} |")

    return "\n".join(lines)
