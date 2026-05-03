"""Integration tests that run actual Hugging Face NER models.

These tests are skipped by default. Run them with `pytest --run-model`.

They require the models to be downloadable (or already cached) and may be slow
on first run.
"""

import pytest

from kuronuri import JA_MODEL, mask, mask_with_label

EN_TEXT = "Hello, I'm Shinsuke Mori. My email address is sincekmori@gmail.com."
JA_TEXT = "こんにちは、森信輔です。私のメールアドレスは sincekmori@gmail.com です。"


@pytest.mark.model
class TestENModelReal:
    def test_block_masks_name_and_email(self) -> None:
        result = mask(EN_TEXT)
        assert "Shinsuke" not in result
        assert "Mori" not in result
        assert "sincekmori" not in result

    def test_block_output(self) -> None:
        result = mask(EN_TEXT)
        assert (
            result
            == "Hello, I'm██████████████. My email address is█████████████████████."
        )

    def test_label_masks_name_and_email(self) -> None:
        result = mask(EN_TEXT, strategy=mask_with_label)
        assert "Shinsuke" not in result
        assert "Mori" not in result
        assert "sincekmori" not in result

    def test_label_output(self) -> None:
        result = mask(EN_TEXT, strategy=mask_with_label)
        assert (
            result == "Hello, I'm<Person><Person>. My email address is<Email><Email>."
        )


@pytest.mark.model
class TestJAModelReal:
    def test_block_masks_name_and_email(self) -> None:
        result = mask(JA_TEXT, model=JA_MODEL)
        assert "森信輔" not in result
        assert "sincekmori" not in result

    def test_block_output(self) -> None:
        result = mask(JA_TEXT, model=JA_MODEL)
        assert (
            result
            == "こんにちは、███です。私のメールアドレスは ██████████@gmail.com です。"
        )

    def test_label_masks_name_and_email(self) -> None:
        result = mask(JA_TEXT, model=JA_MODEL, strategy=mask_with_label)
        assert "森信輔" not in result
        assert "sincekmori" not in result

    def test_label_output(self) -> None:
        result = mask(JA_TEXT, model=JA_MODEL, strategy=mask_with_label)
        assert (
            result
            == "こんにちは、<Person>です。私のメールアドレスは <Person>@gmail.com です。"  # noqa: E501
        )
