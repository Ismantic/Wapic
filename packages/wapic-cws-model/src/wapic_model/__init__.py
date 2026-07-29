"""Access to the default Wapic Chinese word segmentation model."""

from importlib.resources import files


__version__ = "0.1.0"


def model_path():
    """Return the installed default model's filesystem path."""
    path = files(__package__).joinpath("data", "wapic-cws.wac")
    if not path.is_file():
        raise FileNotFoundError(
            "wapic-cws.wac is missing from the wapic-cws-model installation"
        )
    return str(path)


__all__ = ["model_path"]
