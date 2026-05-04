"""Core masking functionality for PII anonymization."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

#: Type alias for a masking strategy function.
MaskStrategy = Callable[[dict[str, Any]], str]


def mask_with_block(entity: dict[str, Any]) -> str:
    """Replace the entity span with █ characters matching its character length."""
    return "█" * (entity["end"] - entity["start"])


def mask_with_label(entity: dict[str, Any]) -> str:
    """Replace the entity with a human-readable label, e.g. ``<Person>``.

    The label is looked up from the entity's ``tag_labels`` key if present
    (injected by :func:`mask`), otherwise the raw ``entity_group`` value is
    wrapped in angle brackets.
    """
    tag = entity["entity_group"]
    label = entity.get("tag_labels", {}).get(tag, tag)
    return f"<{label}>"


def mask_with_fixed(char: str = "*", length: int = 3) -> MaskStrategy:
    """Return a strategy that always substitutes a fixed string.

    Parameters
    ----------
    char:
        The character to repeat.
    length:
        How many times to repeat *char*.

    Examples:
    --------
    >>> strategy = mask_with_fixed(char="*", length=5)
    >>> strategy({"entity_group": "PER", "start": 0, "end": 3, "word": "森信輔"})
    '*****'
    """
    replacement = char * length

    def _strategy(entity: dict[str, Any]) -> str:  # noqa: ARG001
        return replacement

    return _strategy


@dataclass(unsafe_hash=True)
class NERModel:
    """A token-classification model together with its tag vocabulary.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier (any ``token-classification`` model).
    default_mask_tags:
        Tags that :func:`mask` will redact when *mask_tags* is not given.
    tag_labels:
        Optional mapping from raw tag strings to human-readable names used
        by :func:`mask_with_label`.  Tags absent from this mapping fall back
        to the raw tag string.
    aggregation_strategy:
        Pipeline aggregation strategy (default: ``"simple"``).

    Note:
        Hash and equality are determined by ``model_name`` and
        ``aggregation_strategy`` only, so ``tag_labels`` does not affect
        pipeline caching.

    Examples:
    --------
    >>> model = NERModel(
    ...     model_name="my-org/my-ner-model",
    ...     default_mask_tags={"PERSON", "ORG"},
    ...     tag_labels={"PERSON": "Person", "ORG": "Organization"},
    ... )
    """

    model_name: str
    default_mask_tags: frozenset[str]
    tag_labels: dict[str, str] = field(default_factory=dict, hash=False, compare=False)
    aggregation_strategy: str = "simple"


EN_MODEL: NERModel = NERModel(
    model_name="openai/privacy-filter",
    default_mask_tags=frozenset(
        {
            "account_number",
            "private_address",
            "private_date",
            "private_email",
            "private_person",
            "private_phone",
            "private_url",
            "secret",
        }
    ),
    tag_labels={
        "account_number": "AccountNumber",
        "private_address": "Address",
        "private_date": "Date",
        "private_email": "Email",
        "private_person": "Person",
        "private_phone": "Phone",
        "private_url": "URL",
        "secret": "Secret",
    },
)


JA_MODEL: NERModel = NERModel(
    model_name="tsmatz/xlm-roberta-ner-japanese",
    default_mask_tags=frozenset({"PER", "ORG", "LOC"}),
    tag_labels={
        "PER": "Person",
        "ORG": "Organization",
        "ORG-P": "PoliticalOrganization",
        "ORG-O": "OtherOrganization",
        "LOC": "Location",
        "INS": "Institution",
        "PRD": "Product",
        "EVT": "Event",
    },
)


@cache
def _get_pipeline(model: NERModel) -> Any:  # noqa: ANN401
    return hf_pipeline(
        "token-classification",
        model=model.model_name,
        aggregation_strategy=model.aggregation_strategy,
    )


def mask(
    text: str,
    *,
    model: NERModel = EN_MODEL,
    mask_tags: frozenset[str] | set[str] | None = None,
    strategy: MaskStrategy = mask_with_block,
) -> str:
    """Return *text* with detected named entities replaced.

    Parameters
    ----------
    text:
        Input string.
    model:
        :class:`NERModel` instance to use.  Defaults to :data:`EN_MODEL`
        (``openai/privacy-filter``).  Use :data:`JA_MODEL` for Japanese text,
        or supply a custom :class:`NERModel` for any other language.
    mask_tags:
        Set of entity tag strings to redact.  ``None`` (default) uses
        ``model.default_mask_tags``.
    strategy:
        A callable ``(entity: dict) -> str`` that returns the replacement
        string for each detected entity.  Built-in options:

        - :func:`mask_with_block` *(default)* — fills with ``█`` characters
        - :func:`mask_with_label` — replaces with a human-readable label
        - :func:`mask_with_fixed` — factory for fixed char/length replacement

    Returns:
    -------
    str
        The anonymised string.

    Examples:
    --------
    >>> mask("Hello, I'm Shinsuke Mori. My email address is sincekmori@gmail.com.")
    "Hello, I'm██████████████. My email address is█████████████████████."

    >>> mask(
    ...     "Hello, I'm Shinsuke Mori. My email address is sincekmori@gmail.com.",
    ...     strategy=mask_with_label,
    ... )
    "Hello, I'm<Person><Person>. My email address is<Email><Email>."

    >>> mask(
    ...     "こんにちは、森信輔です。私のメールアドレスは sincekmori@gmail.com です。",
    ...     model=JA_MODEL,
    ... )
    'こんにちは、███です。私のメールアドレスは ██████████@gmail.com です。'

    >>> mask(
    ...     "こんにちは、森信輔です。私のメールアドレスは sincekmori@gmail.com です。",
    ...     model=JA_MODEL,
    ...     strategy=mask_with_label,
    ... )
    'こんにちは、<Person>です。私のメールアドレスは <Person>@gmail.com です。'
    """
    if not text:
        return text

    tags = mask_tags if mask_tags is not None else model.default_mask_tags

    pipe = _get_pipeline(model)
    entities: list[dict[str, Any]] = pipe(text)

    # Filter to target tags, then sort descending so end-of-string splices
    # don't invalidate earlier offsets.
    entities_to_mask = sorted(
        (e for e in entities if e["entity_group"] in tags),
        key=lambda e: e["start"],
        reverse=True,
    )

    result = text
    for entity in entities_to_mask:
        # Inject tag_labels so mask_with_label can resolve names without
        # needing a direct reference to the NERModel instance.
        enriched = {**entity, "tag_labels": model.tag_labels}
        result = (
            result[: entity["start"]] + strategy(enriched) + result[entity["end"] :]
        )

    return result
