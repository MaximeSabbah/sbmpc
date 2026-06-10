from __future__ import annotations

import os
from pathlib import Path

import jax


DEFAULT_JAX_CACHE_DIR = Path(__file__).resolve().parents[1] / ".jax_cache"


def configure_jax_compilation_cache(
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Enable JAX's persistent compilation cache for controller kernels."""
    requested = cache_dir
    if requested is None:
        requested = os.environ.get("SBMPC_JAX_CACHE_DIR", DEFAULT_JAX_CACHE_DIR)

    cache_text = os.fspath(requested).strip()
    if cache_text.lower() in {"", "0", "false", "no", "off", "none"}:
        return None

    cache_path = Path(cache_text).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_path))
    jax.config.update("jax_enable_compilation_cache", True)
    if "jax_persistent_cache_min_compile_time_secs" in jax.config.values:
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    if "jax_persistent_cache_min_entry_size_bytes" in jax.config.values:
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return cache_path
