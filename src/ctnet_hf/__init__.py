"""Public package exports and Hugging Face registration helpers."""

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
)

from .configuration_ctnet import CtnetConfig
from .modeling_ctnet import CtnetForEEGClassification, CtnetModel

__all__ = [
    "CtnetConfig",
    "CtnetModel",
    "CtnetForEEGClassification",
]


def _safe_register() -> None:
    try:
        AutoConfig.register("ctnet", CtnetConfig)
    except ValueError:
        pass

    try:
        AutoModel.register(CtnetConfig, CtnetModel)
    except ValueError:
        pass

    try:
        AutoModelForSequenceClassification.register(
            CtnetConfig,
            CtnetForEEGClassification,
        )
    except ValueError:
        pass


_safe_register()
CtnetConfig.register_for_auto_class()
CtnetModel.register_for_auto_class("AutoModel")
CtnetForEEGClassification.register_for_auto_class(
    "AutoModelForSequenceClassification"
)
