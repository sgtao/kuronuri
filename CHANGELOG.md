# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-04

### Changed

- **Breaking:** `MaskStrategy` signature changed from `(entity: dict) -> str` to `(entity: dict, tag_labels: dict[str, str]) -> str`. Custom strategy functions must be updated to accept the new second argument.
- `NERModel` is now hashable (`unsafe_hash=True`), enabling safe use with `functools.lru_cache`. Hash and equality are determined by `model_name` and `aggregation_strategy`; `tag_labels` is excluded from both.
- Pipeline caching switched from a manual `dict` to `@lru_cache`, keyed on the full `NERModel` identity (including `aggregation_strategy`).
- BOM detection in the CLI now returns the BOM bytes directly instead of a `bool` flag, simplifying the read/write path.
- Logo update for improved legibility.

## [0.1.0] - 2026-05-03

### Added

- Initial release of `kuronuri`, a Python library for masking personal information (PII) using Named Entity Recognition (NER) models.
- Core `mask()` function to redact PII from text safely and offline.
- Built-in configurations for English (`EN_MODEL` via `openai/privacy-filter`) and Japanese (`JA_MODEL` via `tsmatz/xlm-roberta-ner-japanese`).
- Support for custom Hugging Face `token-classification` models via the `NERModel` dataclass.
- Three built-in masking strategies:
  - `mask_with_block` (default): Replaces entities with `█` characters of matching length.
  - `mask_with_label`: Replaces entities with human-readable tags (e.g., `<Person>`).
  - `mask_with_fixed`: Replaces entities with a fixed string (e.g., `***`).
- Support for custom masking strategies using user-defined callable functions.
- Command-line interface (CLI) using Typer to mask PII in text files or inline strings.
- CLI file processing that safely preserves original file encodings, BOM (UTF-8/16/32), and line endings (`\n`, `\r\n`, `\r`).
