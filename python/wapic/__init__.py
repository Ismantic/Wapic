import os

from ._core import Segmenter as _CoreSegmenter


class Segmenter(_CoreSegmenter):
    """Load the bundled CWS model, or a user-supplied Wapic model."""

    def __init__(self, model_path=None):
        if model_path is None:
            try:
                from wapic_model import model_path as default_model_path
            except ImportError as exc:
                raise RuntimeError(
                    "The default model is not installed. Run "
                    "`pip install wapic-cws-model` or pass model_path."
                ) from exc
            model_path = default_model_path()
        super().__init__(os.fspath(model_path))


__all__ = ["Segmenter"]
