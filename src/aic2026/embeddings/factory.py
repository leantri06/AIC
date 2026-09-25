"""Share encoders and keep only the active text model on the local GPU."""
from __future__ import annotations

from threading import RLock

from .giant import GIANT_MODEL, SigLIP2GiantTextEmbedder
from .text import OpenCLIPTextEmbedder

_LOCK = RLock()
_ENCODERS: dict[tuple[str, str], ManagedTextEncoder] = {}


def _offload_others(active=None) -> None:
    for encoder in _ENCODERS.values():
        if encoder is not active and encoder.inner.model is not None:
            encoder.inner.model.to("cpu")


class ManagedTextEncoder:
    def __init__(self, inner):
        self.inner = inner

    def load(self) -> None:
        with _LOCK:
            _offload_others(self)
            self.inner.load()
            self.inner.model.to(self.inner.device)

    def encode(self, text: str):
        with _LOCK:
            self.load()
            return self.inner.encode(text)

    def unload(self) -> None:
        with _LOCK:
            # Preserve CPU weights for fast return after another model is used.
            if self.inner.model is not None:
                self.inner.model.to("cpu")


def load_text_encoder(model_name: str, pretrained: str = "webli") -> ManagedTextEncoder:
    if model_name in ("siglip2-giant", GIANT_MODEL):
        model_name = "siglip2-giant"
    elif model_name == "ViT-SO400M-14-SigLIP2-378":
        model_name = "siglip2"
    key = (model_name, pretrained)
    with _LOCK:
        if key not in _ENCODERS:
            _offload_others()
            inner = (SigLIP2GiantTextEmbedder() if model_name == "siglip2-giant"
                     else OpenCLIPTextEmbedder(model_name=model_name, pretrained=pretrained))
            _ENCODERS[key] = ManagedTextEncoder(inner)
        encoder = _ENCODERS[key]
        encoder.load()
        return encoder
