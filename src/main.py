import os
import inspect

def patch_hf_hub_download_use_auth_token() -> None:
    """
    huggingface_hub v1.x fjernet `use_auth_token` og bruker `token` i stedet.
    WhisperX/pyannote i noen versjoner sender fortsatt `use_auth_token`, og da får man:
      TypeError: hf_hub_download() got an unexpected keyword argument 'use_auth_token'

    Denne patchen gjør koden kompatibel uten å endre site-packages.
    """
    try:
        import huggingface_hub
        from huggingface_hub import file_download as hf_file_download
    except Exception:
        return

    # Hvis installert huggingface_hub allerede støtter use_auth_token -> ingenting å gjøre
    try:
        sig = inspect.signature(huggingface_hub.hf_hub_download)
        if "use_auth_token" in sig.parameters:
            return
    except Exception:
        # om signature inspection feiler, prøver vi likevel å patche
        pass

    orig = huggingface_hub.hf_hub_download

    def compat_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs and "token" not in kwargs:
            tok = kwargs.pop("use_auth_token")
            # Noen libs bruker use_auth_token=True for å bruke cached/env token
            if tok is True:
                tok = (
                    os.getenv("HF_TOKEN")
                    or os.getenv("HUGGINGFACE_TOKEN")
                    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                )
            kwargs["token"] = tok
        return orig(*args, **kwargs)

    # Patch top-level og file_download
    huggingface_hub.hf_hub_download = compat_hf_hub_download
    try:
        hf_file_download.hf_hub_download = compat_hf_hub_download
    except Exception:
        pass

    # Patch pyannote sin lokale import (den bruker ofte "from huggingface_hub import hf_hub_download")
    try:
        import pyannote.audio.core.pipeline as pa_pipeline
        pa_pipeline.hf_hub_download = compat_hf_hub_download
    except Exception:
        pass
