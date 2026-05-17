<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/sincekmori/kuronuri@main/logo.svg" width="560" alt="kuronuri — PII redaction library">
</p>

[![PyPI version](https://badge.fury.io/py/kuronuri.svg)](https://badge.fury.io/py/kuronuri)
[![Python Versions](https://img.shields.io/pypi/pyversions/kuronuri.svg)](https://pypi.org/project/kuronuri/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/sincekmori/kuronuri/actions/workflows/ci.yml/badge.svg)](https://github.com/sincekmori/kuronuri/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sincekmori/kuronuri/branch/main/graph/badge.svg)](https://codecov.io/gh/sincekmori/kuronuri)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)

**kuronuri** (黒塗り) is a Python library for masking personal information (PII) in text using Named Entity Recognition (NER) models.

> _kuronuri_ (黒塗り) is the Japanese word for **redaction** — the act of blacking out sensitive information in documents.

## Why kuronuri?

Sending text to an external LLM API is often the most capable approach, but it means personal information leaves your environment.
kuronuri is designed for the use case where you want to **redact PII before passing text to an LLM** (or any other external service), reducing the risk of inadvertent data exposure.

Inference runs **entirely on your machine**: after the model is downloaded on first run, kuronuri works fully offline.
kuronuri never logs, stores, or transmits the text you process.

> [!WARNING]
> NER models are not perfect. kuronuri will miss some entities and may flag false positives.
> Always have a human review the output before treating it as fully anonymised.
> The goal is to reduce the manual redaction burden, not to replace human judgement entirely.

## Features

- 🌐 Built-in models for English (`EN_MODEL`) and Japanese (`JA_MODEL`); any language is supported via any Hugging Face `token-classification` model
- ✏️ Three masking strategies — block fill, human-readable labels, or fixed string
- 🖥️ CLI included — mask files or inline strings from the terminal
- 🤖 MCP server included — expose PII masking as tools to Claude and other MCP clients
- 🐍 Requires Python 3.10 or later

## Installation

**pip:**

```bash
pip install kuronuri
```

**uv:**

```bash
uv add kuronuri
```

### CPU-only environments

If you do not have a GPU, installing the CPU build of PyTorch **before** installing kuronuri avoids downloading the default CUDA build (~200 MB vs ~2 GB).

**pip:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install kuronuri
```

**uv:** First, add the following to your `pyproject.toml`:

```toml
[[tool.uv.index]]
explicit = true
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"

[tool.uv.sources]
torch = {index = "pytorch-cpu"}
```

Then run:

```bash
uv add torch
uv add kuronuri
```

## Quick Start

```python
from kuronuri import mask

# English (default)
mask("Hello, I'm Shinsuke Mori. My email address is sincekmori@gmail.com.")
# → "Hello, I'm██████████████. My email address is█████████████████████."

# Japanese
from kuronuri import JA_MODEL
mask("こんにちは、森信輔です。私のメールアドレスは sincekmori@gmail.com です。", model=JA_MODEL)
# → 'こんにちは、███です。私のメールアドレスは ██████████@gmail.com です。'
```

## Built-in Models

| Constant               | Model                                                                                       | Default language |
| ---------------------- | ------------------------------------------------------------------------------------------- | ---------------- |
| `EN_MODEL` _(default)_ | [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter)                     | English          |
| `JA_MODEL`             | [`tsmatz/xlm-roberta-ner-japanese`](https://huggingface.co/tsmatz/xlm-roberta-ner-japanese) | Japanese         |

### Custom model

Build a `NERModel` for any Hugging Face [`token-classification`](https://huggingface.co/models?pipeline_tag=token-classification) model:

```python
from kuronuri import NERModel, mask

my_model = NERModel(
    model_name="my-org/my-ner-model",
    default_mask_tags={"PERSON", "ORG"},
    tag_labels={"PERSON": "Person", "ORG": "Organization"},
)
mask("...", model=my_model)
```

## Masking Strategies

Three strategies are provided out of the box. You can also pass any callable `(entity: dict) -> str`.

| Strategy                        | Example output | Description                                           |
| ------------------------------- | -------------- | ----------------------------------------------------- |
| `mask_with_block` _(default)_   | `███`          | Fills with `█` matching the entity's character length |
| `mask_with_label`               | `<Person>`     | Replaces with a human-readable label                  |
| `mask_with_fixed(char, length)` | `***`          | Replaces with a fixed string                          |

```python
from kuronuri import mask, mask_with_label, mask_with_fixed

mask("Hello, I'm Shinsuke Mori. My email address is sincekmori@gmail.com.")
# → "Hello, I'm██████████████. My email address is█████████████████████."

mask("Hello, I'm Shinsuke Mori. My email address is sincekmori@gmail.com.", strategy=mask_with_label)
# → "Hello, I'm<Person><Person>. My email address is<Email><Email>."

mask("Hello, I'm Shinsuke Mori. My email address is sincekmori@gmail.com.", strategy=mask_with_fixed(char="*", length=5))
# → "Hello, I'm*****. My email address is*****."

mask("Hello, I'm Shinsuke Mori. My email address is sincekmori@gmail.com.", strategy=lambda e, _labels: f"[{e['entity_group']}]")
# → "Hello, I'm[private_person][private_person]. My email address is[private_email][private_email]."
```

## NER Tags

### `EN_MODEL` — `openai/privacy-filter`

`openai/privacy-filter` is designed specifically for PII detection, so **all of its tags are masked by default**.

| Tag               | `mask_with_label` output | Description               |
| ----------------- | ------------------------ | ------------------------- |
| `private_person`  | `<Person>`               | Person name               |
| `private_address` | `<Address>`              | Physical address          |
| `private_email`   | `<Email>`                | Email address             |
| `private_phone`   | `<Phone>`                | Phone number              |
| `private_url`     | `<URL>`                  | Private URL               |
| `private_date`    | `<Date>`                 | Private date              |
| `account_number`  | `<AccountNumber>`        | Account / card number     |
| `secret`          | `<Secret>`               | API keys, passwords, etc. |

### `JA_MODEL` — `tsmatz/xlm-roberta-ner-japanese`

| Tag     | `mask_with_label` output  | Description            | Masked by default |
| ------- | ------------------------- | ---------------------- | ----------------- |
| `PER`   | `<Person>`                | Person name            | ✅                |
| `ORG`   | `<Organization>`          | General organisation   | ✅                |
| `LOC`   | `<Location>`              | Location               | ✅                |
| `ORG-P` | `<PoliticalOrganization>` | Political organisation | —                 |
| `ORG-O` | `<OtherOrganization>`     | Other organisation     | —                 |
| `INS`   | `<Institution>`           | Institution / facility | —                 |
| `PRD`   | `<Product>`               | Product                | —                 |
| `EVT`   | `<Event>`                 | Event                  | —                 |

Use `mask_tags` to override the default set for any model:

```python
from kuronuri import JA_MODEL, mask

# Mask only person names
mask("こんにちは、森信輔です。私のメールアドレスは sincekmori@gmail.com です。", model=JA_MODEL, mask_tags={"PER"})
# → 'こんにちは、███です。私のメールアドレスは sincekmori@gmail.com です。'
```

## CLI

kuronuri provides two subcommands: `mask` for PII redaction and `serve` for the MCP server.

### `kuronuri mask`

```
Usage: kuronuri mask [OPTIONS] INPUT

  Mask PII in a text file or an inline string.

Arguments:
  INPUT  Path to a text file, or a literal string to mask inline.

Options:
  -o, --output PATH         Output file path. Defaults to stdout.
  -s, --strategy TEXT       Masking strategy: 'block' (███, default),
                            'label' (<Person>), or 'fixed'.
  --fixed-char TEXT         Character used by the 'fixed' strategy.
  --fixed-length INTEGER    Length used by the 'fixed' strategy.  [default: 3]
  -t, --tag TEXT            Entity tag to mask. Repeatable.
      --lang TEXT           Built-in language: 'en' (default) or 'ja'.
                            Mutually exclusive with --model.
  -m, --model TEXT          Hugging Face model identifier for a custom model.
                            Mutually exclusive with --lang.
  -v, --version             Show version and exit.
  --help                    Show this message and exit.
```

**Examples:**

```bash
# Inline string — the 'mask' subcommand can be omitted for convenience
kuronuri "Hello, I'm Shinsuke Mori. My email address is sincekmori@gmail.com."
kuronuri mask "Hello, I'm Shinsuke Mori. My email address is sincekmori@gmail.com."

# Japanese text
kuronuri mask --lang ja "こんにちは、森信輔です。私のメールアドレスは sincekmori@gmail.com です。"

# File → stdout with label strategy
kuronuri mask --strategy label report.txt

# File → output file
kuronuri mask input.txt -o output.txt

# Custom model
kuronuri mask --model my-org/my-ner-model input.txt

# Show version
kuronuri mask --version
```

The CLI preserves the original file encoding (including BOM) and line endings.

## MCP Server

kuronuri includes a built-in [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that exposes PII masking as tools to Claude and other MCP clients.

### Start the server

```bash
kuronuri serve --mcp
```

The server runs over stdio transport, ready to be registered with an MCP client.

### Register with Claude Code

Add the following to your Claude Code MCP configuration (`~/.claude/claude_code_config.json` or `.claude/mcp.json`):

```json
{
  "mcpServers": {
    "kuronuri": {
      "command": "kuronuri",
      "args": ["serve", "--mcp"]
    }
  }
}
```

### Register with Claude Desktop

Add the following to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "kuronuri": {
      "command": "kuronuri",
      "args": ["serve", "--mcp"]
    }
  }
}
```

### Available MCP tools

#### `mask_text`

Mask PII in the given text and return the result.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | — | Input text to mask |
| `lang` | `str` | `"en"` | Language code: `"en"` or `"ja"` |
| `strategy` | `str` | `"block"` | Masking strategy: `"block"` / `"label"` / `"fixed"` |
| `mask_tags` | `list[str] \| null` | `null` | NER tags to mask. `null` uses the model default |
| `fixed_char` | `str` | `"█"` | Replacement character for the `"fixed"` strategy |
| `fixed_length` | `int` | `3` | Number of replacement characters for the `"fixed"` strategy |

#### `list_ner_tags`

Return a Markdown table of NER tags and default mask targets for the given language.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lang` | `str` | `"en"` | Language code: `"en"` or `"ja"` |

## API Reference

### `mask(text, *, model, mask_tags, strategy) -> str`

| Parameter   | Type                                    | Default           | Description                                          |
| ----------- | --------------------------------------- | ----------------- | ---------------------------------------------------- |
| `text`      | `str`                                   | —                 | Input string                                         |
| `model`     | `NERModel`                              | `EN_MODEL`        | NER model to use                                     |
| `mask_tags` | `set[str] \| None`                      | `None`            | Tags to mask. `None` uses `model.default_mask_tags`. |
| `strategy`  | `Callable[[dict, dict[str, str]], str]` | `mask_with_block` | Masking strategy                                     |

### `NERModel`

```python
@dataclass
class NERModel:
    model_name: str                   # Hugging Face model identifier
    default_mask_tags: frozenset[str] # tags redacted when mask_tags=None
    tag_labels: dict[str, str]        # tag → label for mask_with_label
    aggregation_strategy: str         # default: "simple"
```

### Built-in strategies

| Function                        | Description                             |
| ------------------------------- | --------------------------------------- |
| `mask_with_block`               | `█` × entity character length (default) |
| `mask_with_label`               | `<Person>` style label                  |
| `mask_with_fixed(char, length)` | Factory for a fixed replacement string  |
