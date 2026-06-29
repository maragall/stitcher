"""Registration channel is the operator's explicit choice -- there is NO auto-pick.

Contrast/entropy proxies don't reliably identify the best-registering channel across
datasets (and std alone picks a saturated, useless channel on brightfield), so an
automatic guess can silently misalign the mosaic. We therefore require channel_to_use
for multi-channel data (cf. ASHLAR --align-channel).
"""

import pytest

from tilefusion.core import TileFusion


class _Fake:
    """Minimal stand-in exposing only what the channel-resolution paths touch."""

    def __init__(self, channels, channel_to_use):
        self.channels = channels
        self.channel_to_use = channel_to_use


# --------------------------------------------------------------------------- #
# No auto-pick: explicit channel required for multi-channel data.
# --------------------------------------------------------------------------- #
def test_single_channel_resolves_to_zero():
    f = _Fake(channels=1, channel_to_use=None)
    TileFusion._resolve_registration_channel(f)
    assert f.channel_to_use == 0


def test_multichannel_none_is_deferred_not_autopicked():
    # Stays None at construction (metadata-only uses still work); the requirement is
    # enforced at registration time, NOT silently auto-picked.
    f = _Fake(channels=4, channel_to_use=None)
    TileFusion._resolve_registration_channel(f)
    assert f.channel_to_use is None


def test_explicit_channel_kept():
    f = _Fake(channels=4, channel_to_use=2)
    TileFusion._resolve_registration_channel(f)
    assert f.channel_to_use == 2


def test_out_of_range_channel_raises():
    f = _Fake(channels=4, channel_to_use=9)
    with pytest.raises(ValueError):
        TileFusion._resolve_registration_channel(f)


def test_registration_without_channel_raises():
    # The backstop: if no channel reaches registration, fail loudly with guidance
    # rather than guessing.
    f = _Fake(channels=4, channel_to_use=None)
    with pytest.raises(ValueError, match="registration channel"):
        TileFusion.refine_tile_positions_with_cross_correlation(f)
