"""Tests for kuronuri._cli."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from kuronuri._cli import app
from kuronuri._masker import EN_MODEL, JA_MODEL, NERModel

runner = CliRunner()


def _mock_mask(text: str, **kwargs: object) -> str:  # noqa: ARG001
    return f"[{text}]"


class TestVersion:
    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version", "dummy"])
        assert result.exit_code == 0
        assert "kuronuri" in result.output


class TestInlineMode:
    def test_inline_text_masked(self) -> None:
        with patch("kuronuri._cli.mask", side_effect=_mock_mask):
            result = runner.invoke(app, ["hello"])
        assert result.exit_code == 0
        assert "[hello]" in result.output

    def test_no_flags_uses_en_model(self) -> None:
        captured: dict = {}

        def _capture(text: str, **kwargs: object) -> str:
            captured.update(kwargs)
            return text

        with patch("kuronuri._cli.mask", side_effect=_capture):
            runner.invoke(app, ["test"])
        assert captured["model"] is EN_MODEL

    def test_lang_en_uses_en_model(self) -> None:
        captured: dict = {}

        def _capture(text: str, **kwargs: object) -> str:
            captured.update(kwargs)
            return text

        with patch("kuronuri._cli.mask", side_effect=_capture):
            runner.invoke(app, ["--lang", "en", "test"])
        assert captured["model"] is EN_MODEL

    def test_lang_ja_uses_ja_model(self) -> None:
        captured: dict = {}

        def _capture(text: str, **kwargs: object) -> str:
            captured.update(kwargs)
            return text

        with patch("kuronuri._cli.mask", side_effect=_capture):
            runner.invoke(app, ["--lang", "ja", "テスト"])
        assert captured["model"] is JA_MODEL

    def test_custom_model_builds_ner_model(self) -> None:
        captured: dict = {}

        def _capture(text: str, **kwargs: object) -> str:
            captured.update(kwargs)
            return text

        with patch("kuronuri._cli.mask", side_effect=_capture):
            runner.invoke(app, ["--model", "my-org/my-model", "test"])
        assert isinstance(captured["model"], NERModel)
        assert captured["model"].model_name == "my-org/my-model"

    def test_lang_and_model_both_given_exits_with_error(self) -> None:
        result = runner.invoke(app, ["--lang", "en", "--model", "org/m", "test"])
        assert result.exit_code != 0

    def test_unknown_lang_exits_with_error(self) -> None:
        result = runner.invoke(app, ["--lang", "zz", "test"])
        assert result.exit_code != 0

    def test_strategy_block(self) -> None:
        captured: dict = {}

        def _capture(text: str, **kwargs: object) -> str:
            captured.update(kwargs)
            return text

        with patch("kuronuri._cli.mask", side_effect=_capture):
            runner.invoke(app, ["--strategy", "block", "test"])
        from kuronuri._masker import mask_with_block  # noqa: PLC0415

        assert captured["strategy"] is mask_with_block

    def test_strategy_label(self) -> None:
        captured: dict = {}

        def _capture(text: str, **kwargs: object) -> str:
            captured.update(kwargs)
            return text

        with patch("kuronuri._cli.mask", side_effect=_capture):
            runner.invoke(app, ["--strategy", "label", "test"])
        from kuronuri._masker import mask_with_label  # noqa: PLC0415

        assert captured["strategy"] is mask_with_label

    def test_strategy_fixed_default_params(self) -> None:
        captured: dict = {}

        def _capture(text: str, **kwargs: object) -> str:
            captured.update(kwargs)
            return text

        with patch("kuronuri._cli.mask", side_effect=_capture):
            runner.invoke(app, ["--strategy", "fixed", "test"])
        entity = {"entity_group": "PER", "start": 0, "end": 2, "word": "AB"}
        assert captured["strategy"](entity, {}) == "***"

    def test_custom_tags_passed_as_mask_tags(self) -> None:
        captured: dict = {}

        def _capture(text: str, **kwargs: object) -> str:
            captured.update(kwargs)
            return text

        with patch("kuronuri._cli.mask", side_effect=_capture):
            runner.invoke(app, ["--tag", "PER", "--tag", "LOC", "test"])
        assert captured["mask_tags"] == frozenset({"PER", "LOC"})

    def test_no_tags_flag_passes_none(self) -> None:
        captured: dict = {}

        def _capture(text: str, **kwargs: object) -> str:
            captured.update(kwargs)
            return text

        with patch("kuronuri._cli.mask", side_effect=_capture):
            runner.invoke(app, ["test"])
        assert captured["mask_tags"] is None


class TestFileMode:
    def test_file_masked_to_stdout(self, tmp_path: Path) -> None:
        src = tmp_path / "input.txt"
        src.write_text("Hello Alice.\n", encoding="utf-8")

        with patch("kuronuri._cli.mask", side_effect=_mock_mask):
            result = runner.invoke(app, [str(src)])
        assert result.exit_code == 0
        assert "[" in result.output

    def test_file_masked_to_output_file(self, tmp_path: Path) -> None:
        src = tmp_path / "input.txt"
        src.write_text("Hello Alice.\n", encoding="utf-8")
        dst = tmp_path / "output.txt"

        with patch("kuronuri._cli.mask", side_effect=_mock_mask):
            result = runner.invoke(app, [str(src), "--output", str(dst)])
        assert result.exit_code == 0
        assert dst.exists()

    def test_utf8_bom_preserved(self, tmp_path: Path) -> None:
        src = tmp_path / "input.txt"
        content = "Hello Alice.\n"
        src.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

        with patch("kuronuri._cli.mask", return_value=content):
            result = runner.invoke(app, [str(src)])
        assert result.exit_code == 0

    def test_utf16le_bom_preserved(self, tmp_path: Path) -> None:
        import codecs  # noqa: PLC0415

        src = tmp_path / "input.txt"
        content = "Hello Alice.\n"
        src.write_bytes(codecs.BOM_UTF16_LE + content.encode("utf-16-le"))

        with patch("kuronuri._cli.mask", return_value=content):
            result = runner.invoke(app, [str(src)])
        assert result.exit_code == 0

    @pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
    def test_newline_preserved(self, tmp_path: Path, newline: str) -> None:
        src = tmp_path / "input.txt"
        raw = newline.join(["line one", "line two", "line three"]).encode("utf-8")
        src.write_bytes(raw)

        with patch("kuronuri._cli.mask", side_effect=lambda t, **kw: t):  # noqa: ARG005
            result = runner.invoke(app, [str(src)])
        assert result.exit_code == 0

    def test_non_utf8_file_exits_with_error(self, tmp_path: Path) -> None:
        src = tmp_path / "input.txt"
        src.write_bytes("鈴木".encode("shift_jis"))

        result = runner.invoke(app, [str(src)])
        assert result.exit_code == 1


class TestInvalidStrategy:
    def test_unknown_strategy_raises(self) -> None:
        result = runner.invoke(app, ["--strategy", "nonexistent", "test"])
        assert result.exit_code != 0
