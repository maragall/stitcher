"""
Shared helpers for Squid-format folder readers.

Both ``ome_tiff_tiles`` and ``individual_tiffs`` readers share:
  - ``load_acquisition_params`` — reads ``acquisition parameters.json``
  - ``channel_names_or_default`` — fills in synthetic names when OME metadata is absent

The coordinates.csv parsers differ between the two formats (different CSV paths,
different column conventions, different z-level filtering), so they are kept
loader-specific rather than forced into a single shared helper.  See the note in
each loader for the exact divergence.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_acquisition_params(folder: Path) -> tuple[float, int, int, float]:
    """
    Read ``acquisition parameters.json`` from *folder* and return
    ``(pixel_size_um, n_z, n_t, dz_um)``.

    Fallbacks (when the file is absent or a key is missing):
      - magnification  : 10.0
      - sensor_pixel   : 7.52 µm  →  pixel_size_um = 0.752
      - Nz             : 1
      - Nt             : 1
      - dz(um)         : 1.0

    These match the inline defaults used by both
    ``load_ome_tiff_tiles_metadata`` and ``load_individual_tiffs_metadata``
    verbatim.

    Parameters
    ----------
    folder:
        Dataset root folder (the one that *may* contain
        ``acquisition parameters.json``).

    Returns
    -------
    pixel_size_um, n_z, n_t, dz_um
    """
    params_path = Path(folder) / "acquisition parameters.json"
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        magnification = params.get("objective", {}).get("magnification", 10.0)
        sensor_pixel_um = params.get("sensor_pixel_size_um", 7.52)
        pixel_size_um = sensor_pixel_um / magnification
        n_z = params.get("Nz", 1)
        n_t = params.get("Nt", 1)
        dz_um = params.get("dz(um)", 1.0)
    else:
        pixel_size_um = 0.752
        n_z = 1
        n_t = 1
        dz_um = 1.0
    return pixel_size_um, n_z, n_t, dz_um


def channel_names_or_default(names: list[str], channels: int) -> list[str]:
    """
    Return *names* if non-empty, otherwise generate ``["Channel_0", ...]``.

    Parameters
    ----------
    names:
        List of channel names (may be empty).
    channels:
        Number of channels — used only when *names* is empty.

    Returns
    -------
    list[str]
        Non-empty list of channel name strings.
    """
    if names:
        return names
    return [f"Channel_{i}" for i in range(channels)]
