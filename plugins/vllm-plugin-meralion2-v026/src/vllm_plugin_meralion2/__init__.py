"""vLLM plugin entrypoint and registration for MERaLiON2 models."""

from packaging.version import Version


def _patch_logitsprocs_output_token_tracking() -> None:
    """Fix vLLM bug: entry-point logits processors don't enable output token tracking.

    vLLM sets ``logitsprocs_need_output_token_ids`` based on CLI-passed
    ``custom_logitsprocs`` only, ignoring processors discovered via the
    ``vllm.logits_processors`` entry-point group.  When this flag is False
    and ``repetition_penalty=1.0`` (no penalties), the output-token live
    reference is filled with ``-1`` placeholders instead of actual token IDs,
    silently disabling any processor that inspects generation history
    (e.g. ``NoRepeatNGramV1LogitsProcessor``).

    This patch wraps ``InputBatch.__init__`` to force the flag when
    non-argmax-invariant entry-point processors are loaded.
    """
    try:
        from vllm.v1.worker.gpu_input_batch import InputBatch
    except ImportError:
        return  # vLLM version without V1 InputBatch

    _orig_init = InputBatch.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # If entry-point plugins loaded non-argmax-invariant processors,
        # they need output_token_ids to be tracked — force the flag.
        if (hasattr(self, "logitsprocs")
                and self.logitsprocs.non_argmax_invariant
                and not self.logitsprocs_need_output_token_ids):
            self.logitsprocs_need_output_token_ids = True

    InputBatch.__init__ = _patched_init


def register() -> None:
    """Register MERaLiON2 model with vLLM's plugin system.

    Supported vLLM versions: >= 0.26.0, < 0.27.0.

    This fork targets the vLLM 0.26 V1 multimodal embedding contract. It is
    intentionally isolated from the published 0.3 plugin, which supports
    vLLM < 0.17.

    Raises:
        RuntimeError: If the installed vLLM version is not supported.
    """
    from importlib import import_module

    import vllm
    from vllm import ModelRegistry

    current_version = Version(vllm.__version__)
    min_supported_version = Version("0.26.0")
    # The InputBatch patch is retained because vLLM 0.26 still determines
    # token tracking from CLI processors rather than entry-point processors.
    max_supported_version = Version("0.27.0")

    if min_supported_version > current_version or current_version >= max_supported_version:
        raise RuntimeError(
            f"MERaLiON2 plugin does not support vLLM version {vllm.__version__}. "
            f"Supported range: >= {min_supported_version}, < {max_supported_version}"
        )

    module = import_module("vllm_plugin_meralion2.vllm0101")
    meralion2_model_cls = getattr(module, "MERaLiON2ForConditionalGeneration")

    if "MERaLiON2ForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "MERaLiON2ForConditionalGeneration",
            meralion2_model_cls,
        )

    _patch_logitsprocs_output_token_tracking()
