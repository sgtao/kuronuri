"""Tests for kuronuri._masker."""

from unittest.mock import MagicMock, patch

from kuronuri import (
    EN_MODEL,
    JA_MODEL,
    NERModel,
    mask,
    mask_with_block,
    mask_with_fixed,
    mask_with_label,
)
from kuronuri._masker import _get_pipeline, _pipeline_cache


class TestMaskWithBlock:
    def test_fills_with_full_width_squares(self) -> None:
        entity = {"entity_group": "PER", "start": 0, "end": 2, "word": "鈴木"}
        assert mask_with_block(entity) == "██"

    def test_length_matches_span(self) -> None:
        entity = {"entity_group": "ORG", "start": 3, "end": 6, "word": "ABC"}
        assert mask_with_block(entity) == "███"

    def test_zero_length(self) -> None:
        entity = {"entity_group": "PER", "start": 5, "end": 5, "word": ""}
        assert mask_with_block(entity) == ""


class TestMaskWithLabel:
    def test_resolves_from_tag_labels(self) -> None:
        entity = {
            "entity_group": "PER",
            "start": 0,
            "end": 2,
            "word": "鈴木",
            "tag_labels": {"PER": "Person"},
        }
        assert mask_with_label(entity) == "<Person>"

    def test_falls_back_to_raw_tag_when_no_tag_labels(self) -> None:
        entity = {"entity_group": "PER", "start": 0, "end": 2, "word": "鈴木"}
        assert mask_with_label(entity) == "<PER>"

    def test_falls_back_to_raw_tag_when_key_missing(self) -> None:
        entity = {
            "entity_group": "CUSTOM",
            "start": 0,
            "end": 2,
            "word": "XX",
            "tag_labels": {"PER": "Person"},
        }
        assert mask_with_label(entity) == "<CUSTOM>"


class TestMaskWithFixed:
    def test_default_char_and_length(self) -> None:
        strategy = mask_with_fixed()
        entity = {"entity_group": "PER", "start": 0, "end": 10, "word": "something"}
        assert strategy(entity) == "***"

    def test_custom_char_and_length(self) -> None:
        strategy = mask_with_fixed(char="X", length=5)
        entity = {"entity_group": "PER", "start": 0, "end": 2, "word": "鈴木"}
        assert strategy(entity) == "XXXXX"

    def test_returns_callable(self) -> None:
        assert callable(mask_with_fixed())


class TestENModel:
    def test_model_name(self) -> None:
        assert EN_MODEL.model_name == "openai/privacy-filter"

    def test_all_tags_masked_by_default(self) -> None:
        expected = {
            "account_number",
            "private_address",
            "private_date",
            "private_email",
            "private_person",
            "private_phone",
            "private_url",
            "secret",
        }
        assert EN_MODEL.default_mask_tags == expected

    def test_tag_labels(self) -> None:
        assert EN_MODEL.tag_labels["private_person"] == "Person"
        assert EN_MODEL.tag_labels["secret"] == "Secret"  # noqa: S105
        assert EN_MODEL.tag_labels["private_url"] == "URL"

    def test_all_tag_labels_present(self) -> None:
        assert set(EN_MODEL.tag_labels.keys()) == EN_MODEL.default_mask_tags


class TestJAModel:
    def test_model_name(self) -> None:
        assert JA_MODEL.model_name == "tsmatz/xlm-roberta-ner-japanese"

    def test_default_mask_tags(self) -> None:
        assert JA_MODEL.default_mask_tags == frozenset({"PER", "ORG", "LOC"})

    def test_tag_labels(self) -> None:
        assert JA_MODEL.tag_labels["PER"] == "Person"
        assert JA_MODEL.tag_labels["LOC"] == "Location"


class TestNERModel:
    def test_custom_model(self) -> None:
        custom = NERModel(
            model_name="my-org/my-model",
            default_mask_tags=frozenset({"PERSON", "ORG"}),
            tag_labels={"PERSON": "Person", "ORG": "Organization"},
        )
        assert custom.model_name == "my-org/my-model"
        assert "PERSON" in custom.default_mask_tags
        assert custom.tag_labels["PERSON"] == "Person"

    def test_empty_tag_labels_by_default(self) -> None:
        model = NERModel(model_name="some/model", default_mask_tags=frozenset({"X"}))
        assert model.tag_labels == {}

    def test_no_register_lang_exported(self) -> None:
        import kuronuri  # noqa: PLC0415

        assert not hasattr(kuronuri, "register_lang")
        assert not hasattr(kuronuri, "resolve_lang")
        assert not hasattr(kuronuri, "DEFAULT_LANG")


def _make_pipe(entities: list[dict]) -> MagicMock:
    return MagicMock(return_value=entities)


class TestMask:
    def _patch(self, entities: list[dict]) -> patch:  # ty:ignore[invalid-type-form]
        return patch(
            "kuronuri._masker._get_pipeline", return_value=_make_pipe(entities)
        )

    def test_empty_string_returns_empty(self) -> None:
        assert mask("") == ""

    def test_no_args_uses_en_model(self) -> None:
        with patch(
            "kuronuri._masker._get_pipeline", return_value=_make_pipe([])
        ) as mock_get:
            mask("test")
            mock_get.assert_called_once_with(EN_MODEL)

    def test_ja_model(self) -> None:
        with patch(
            "kuronuri._masker._get_pipeline", return_value=_make_pipe([])
        ) as mock_get:
            mask("テスト", model=JA_MODEL)
            mock_get.assert_called_once_with(JA_MODEL)

    def test_default_strategy_is_block(self) -> None:
        entities = [
            {"entity_group": "private_person", "start": 11, "end": 16, "word": "Alice"}
        ]
        with self._patch(entities):
            assert mask("My name is Alice.") == "My name is █████."

    def test_label_strategy_en_model(self) -> None:
        entities = [
            {"entity_group": "private_person", "start": 11, "end": 16, "word": "Alice"}
        ]
        with self._patch(entities):
            assert mask("My name is Alice.", strategy=mask_with_label) == (
                "My name is <Person>."
            )

    def test_label_strategy_ja_model(self) -> None:
        entities = [{"entity_group": "PER", "start": 0, "end": 2, "word": "鈴木"}]
        with self._patch(entities):
            assert mask("鈴木さん", model=JA_MODEL, strategy=mask_with_label) == (
                "<Person>さん"
            )

    def test_en_model_masks_all_tags_by_default(self) -> None:
        # secret is now in EN_MODEL.default_mask_tags
        entities = [
            {"entity_group": "secret", "start": 5, "end": 13, "word": "sk-abc123"}
        ]
        with self._patch(entities):
            result = mask("key: sk-abc123")
            assert "sk-abc123" not in result

    def test_en_model_masks_url_by_default(self) -> None:
        entities = [
            {
                "entity_group": "private_url",
                "start": 6,
                "end": 24,
                "word": "http://example.com",
            }
        ]
        with self._patch(entities):
            result = mask("Visit http://example.com")
            assert "http://example.com" not in result

    def test_mask_tags_overrides_model_default(self) -> None:
        entities = [{"entity_group": "PER", "start": 0, "end": 2, "word": "鈴木"}]
        with self._patch(entities):
            # Only mask LOC, not PER
            result = mask("鈴木さん", model=JA_MODEL, mask_tags={"LOC"})
            assert "鈴木" in result

    def test_mask_tags_none_uses_model_default(self) -> None:
        entities = [{"entity_group": "PRD", "start": 0, "end": 5, "word": "Tokio"}]
        with self._patch(entities):
            # PRD not in JA_MODEL.default_mask_tags
            result = mask("Tokio is great", model=JA_MODEL, mask_tags=None)
            assert "Tokio" in result

    def test_multiple_entities(self) -> None:
        entities = [
            {"entity_group": "PER", "start": 0, "end": 2, "word": "鈴木"},
            {"entity_group": "ORG", "start": 5, "end": 8, "word": "トヨタ"},
        ]
        with self._patch(entities):
            result = mask("鈴木さんはトヨタで", model=JA_MODEL)
            assert "鈴木" not in result
            assert "トヨタ" not in result

    def test_custom_model_passed_to_get_pipeline(self) -> None:
        custom = NERModel(model_name="my-org/my-model", default_mask_tags=frozenset())
        with patch(
            "kuronuri._masker._get_pipeline", return_value=_make_pipe([])
        ) as mock_get:
            mask("テスト", model=custom)
            mock_get.assert_called_once_with(custom)


class TestPipelineCache:
    def test_same_model_cached(self) -> None:
        _pipeline_cache.clear()
        model = NERModel(model_name="model-a", default_mask_tags=frozenset())
        with patch("kuronuri._masker.hf_pipeline") as mock_hf:
            mock_hf.return_value = MagicMock(return_value=[])
            _get_pipeline(model)
            _get_pipeline(model)
            assert mock_hf.call_count == 1

    def test_different_models_separate_cache(self) -> None:
        _pipeline_cache.clear()
        model_a = NERModel(model_name="model-a", default_mask_tags=frozenset())
        model_b = NERModel(model_name="model-b", default_mask_tags=frozenset())
        with patch("kuronuri._masker.hf_pipeline") as mock_hf:
            mock_hf.return_value = MagicMock(return_value=[])
            _get_pipeline(model_a)
            _get_pipeline(model_b)
            assert mock_hf.call_count == 2  # noqa: PLR2004

    def test_same_model_name_shares_cache(self) -> None:
        _pipeline_cache.clear()
        model_a = NERModel(
            model_name="model-x",
            default_mask_tags=frozenset(),
            aggregation_strategy="simple",
        )
        model_b = NERModel(
            model_name="model-x",
            default_mask_tags=frozenset(),
            aggregation_strategy="first",
        )
        with patch("kuronuri._masker.hf_pipeline") as mock_hf:
            mock_hf.return_value = MagicMock(return_value=[])
            _get_pipeline(model_a)
            _get_pipeline(model_b)
            assert mock_hf.call_count == 1
