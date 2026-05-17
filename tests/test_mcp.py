"""Tests for kuronuri._mcp MCP tools."""

from unittest.mock import MagicMock, patch

import pytest

from kuronuri._mcp import list_ner_tags, mask_text


@pytest.fixture
def mock_masker():
    """Provide mocked masker objects to avoid loading real NER models."""
    en_model = MagicMock(name="EN_MODEL")
    ja_model = MagicMock(name="JA_MODEL")
    mask_with_block_fn = MagicMock(name="mask_with_block")
    mask_with_label_fn = MagicMock(name="mask_with_label")
    mask_with_fixed_strategy = MagicMock(name="mask_with_fixed_strategy")
    mask_with_fixed_fn = MagicMock(return_value=mask_with_fixed_strategy)
    mask_fn = MagicMock(return_value="masked text")

    with (
        patch("kuronuri._mcp.EN_MODEL", en_model),
        patch("kuronuri._mcp.JA_MODEL", ja_model),
        patch("kuronuri._mcp.mask", mask_fn),
        patch("kuronuri._mcp.mask_with_block", mask_with_block_fn),
        patch("kuronuri._mcp.mask_with_label", mask_with_label_fn),
        patch("kuronuri._mcp.mask_with_fixed", mask_with_fixed_fn),
    ):
        yield {
            "en_model": en_model,
            "ja_model": ja_model,
            "mask": mask_fn,
            "mask_with_block": mask_with_block_fn,
            "mask_with_label": mask_with_label_fn,
            "mask_with_fixed": mask_with_fixed_fn,
            "mask_with_fixed_strategy": mask_with_fixed_strategy,
        }


@pytest.fixture
def mock_models():
    """Provide mocked NER model objects with tag metadata."""
    en_model = MagicMock(name="EN_MODEL")
    en_model.model_name = "test-en-model"
    en_model.tag_labels = {"PER": "Person", "ORG": "Organization", "LOC": "Location"}
    en_model.default_mask_tags = frozenset({"PER", "ORG"})

    ja_model = MagicMock(name="JA_MODEL")
    ja_model.model_name = "test-ja-model"
    ja_model.tag_labels = {"PER": "Person", "ORG": "Organization"}
    ja_model.default_mask_tags = frozenset({"PER"})

    with (
        patch("kuronuri._mcp.EN_MODEL", en_model),
        patch("kuronuri._mcp.JA_MODEL", ja_model),
    ):
        yield {"en_model": en_model, "ja_model": ja_model}


class TestMaskText:
    def test_default_strategy_uses_block(self, mock_masker: dict) -> None:
        mask_text("Hello John", lang="en")

        mock_masker["mask"].assert_called_once_with(
            "Hello John",
            model=mock_masker["en_model"],
            mask_tags=None,
            strategy=mock_masker["mask_with_block"],
        )

    def test_label_strategy(self, mock_masker: dict) -> None:
        mask_text("Hello John", lang="en", strategy="label")

        mock_masker["mask"].assert_called_once_with(
            "Hello John",
            model=mock_masker["en_model"],
            mask_tags=None,
            strategy=mock_masker["mask_with_label"],
        )

    def test_fixed_strategy(self, mock_masker: dict) -> None:
        mask_text(
            "Hello John", lang="en", strategy="fixed", fixed_char="*", fixed_length=4
        )

        mock_masker["mask_with_fixed"].assert_called_once_with(char="*", length=4)
        mock_masker["mask"].assert_called_once_with(
            "Hello John",
            model=mock_masker["en_model"],
            mask_tags=None,
            strategy=mock_masker["mask_with_fixed_strategy"],
        )

    def test_ja_lang_uses_ja_model(self, mock_masker: dict) -> None:
        mask_text("山田太郎です", lang="ja")

        mock_masker["mask"].assert_called_once_with(
            "山田太郎です",
            model=mock_masker["ja_model"],
            mask_tags=None,
            strategy=mock_masker["mask_with_block"],
        )

    def test_with_mask_tags(self, mock_masker: dict) -> None:
        mask_text("Hello John", lang="en", mask_tags=["PER", "ORG"])

        mock_masker["mask"].assert_called_once_with(
            "Hello John",
            model=mock_masker["en_model"],
            mask_tags=frozenset({"PER", "ORG"}),
            strategy=mock_masker["mask_with_block"],
        )

    def test_no_mask_tags_passes_none(self, mock_masker: dict) -> None:
        mask_text("Hello John", lang="en")

        mock_masker["mask"].assert_called_once_with(
            "Hello John",
            model=mock_masker["en_model"],
            mask_tags=None,
            strategy=mock_masker["mask_with_block"],
        )

    def test_returns_string(self, mock_masker: dict) -> None:
        result = mask_text("some text")

        assert isinstance(result, str)

    def test_unknown_lang_falls_back_to_en_model(self, mock_masker: dict) -> None:
        mask_text("text", lang="fr")

        mock_masker["mask"].assert_called_once_with(
            "text",
            model=mock_masker["en_model"],
            mask_tags=None,
            strategy=mock_masker["mask_with_block"],
        )

    @pytest.mark.model
    def test_en_real_model(self) -> None:
        result = mask_text("My name is John Smith and I live in New York.", lang="en")

        assert isinstance(result, str)
        assert "John Smith" not in result

    @pytest.mark.model
    def test_ja_real_model(self) -> None:
        result = mask_text("私は山田太郎です。", lang="ja")

        assert isinstance(result, str)
        assert "山田太郎" not in result


class TestListNerTags:
    def test_returns_string(self, mock_models: dict) -> None:
        result = list_ner_tags(lang="en")

        assert isinstance(result, str)

    def test_en_contains_model_name(self, mock_models: dict) -> None:
        result = list_ner_tags(lang="en")

        assert "test-en-model" in result

    def test_en_contains_all_tags(self, mock_models: dict) -> None:
        result = list_ner_tags(lang="en")

        assert "PER" in result
        assert "ORG" in result
        assert "LOC" in result

    def test_en_marks_default_mask_tags(self, mock_models: dict) -> None:
        result = list_ner_tags(lang="en")

        lines = result.splitlines()
        per_line = next(line for line in lines if "PER" in line and "|" in line)
        loc_line = next(line for line in lines if "LOC" in line)

        assert "✅" in per_line
        assert "—" in loc_line

    def test_ja_uses_ja_model(self, mock_models: dict) -> None:
        result = list_ner_tags(lang="ja")

        assert "test-ja-model" in result

    def test_default_lang_is_en(self, mock_models: dict) -> None:
        result = list_ner_tags()

        assert "test-en-model" in result

    def test_output_is_markdown_table(self, mock_models: dict) -> None:
        result = list_ner_tags(lang="en")

        assert "| Tag |" in result
        assert "|---|" in result

    @pytest.mark.model
    def test_en_real_model(self) -> None:
        result = list_ner_tags(lang="en")

        assert isinstance(result, str)
        assert "## EN" in result
        assert "| Tag |" in result

    @pytest.mark.model
    def test_ja_real_model(self) -> None:
        result = list_ner_tags(lang="ja")

        assert isinstance(result, str)
        assert "## JA" in result
