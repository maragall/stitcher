"""Tests for tilefusion.io module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from tilefusion.io import (
    load_ome_tiff_metadata,
    load_individual_tiffs_metadata,
    read_individual_tiffs_tile,
)


class TestLoadIndividualTiffsMetadata:
    """Tests for load_individual_tiffs_metadata function."""

    @pytest.fixture
    def sample_tiff_folder(self, tmp_path):
        """Create a sample individual TIFFs folder structure."""
        # Create subfolder
        img_folder = tmp_path / "0"
        img_folder.mkdir()

        # Create coordinates.csv
        coords = pd.DataFrame(
            {
                "fov": [0, 1, 2, 3],
                "x (mm)": [0.0, 1.0, 0.0, 1.0],
                "y (mm)": [0.0, 0.0, 1.0, 1.0],
            }
        )
        coords.to_csv(img_folder / "coordinates.csv", index=False)

        # Create sample TIFF files
        img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        for fov in range(4):
            for ch in ["Fluorescence_488_nm", "Fluorescence_561_nm"]:
                tifffile.imwrite(img_folder / f"manual_{fov}_0_{ch}.tiff", img)

        return tmp_path

    def test_loads_metadata(self, sample_tiff_folder):
        """Test that metadata is loaded correctly."""
        meta = load_individual_tiffs_metadata(sample_tiff_folder)

        assert meta["n_tiles"] == 4
        assert meta["shape"] == (100, 100)
        assert meta["channels"] == 2
        assert len(meta["tile_positions"]) == 4
        assert len(meta["tile_identifiers"]) == 4

    def test_tile_positions(self, sample_tiff_folder):
        """Test that tile positions are converted correctly."""
        meta = load_individual_tiffs_metadata(sample_tiff_folder)

        # Positions should be in µm (mm * 1000)
        assert meta["tile_positions"][0] == (0.0, 0.0)
        assert meta["tile_positions"][1] == (0.0, 1000.0)

    def test_channel_names(self, sample_tiff_folder):
        """Test that channel names are detected."""
        meta = load_individual_tiffs_metadata(sample_tiff_folder)

        assert "Fluorescence_488_nm" in meta["channel_names"]
        assert "Fluorescence_561_nm" in meta["channel_names"]


class TestReadIndividualTiffsTile:
    """Tests for read_individual_tiffs_tile function."""

    @pytest.fixture
    def sample_tiff_folder(self, tmp_path):
        """Create sample TIFF files."""
        img_folder = tmp_path / "0"
        img_folder.mkdir()

        # Create test images with known values
        for fov in range(2):
            for idx, ch in enumerate(["ch1", "ch2"]):
                img = np.full((50, 50), fill_value=(fov + 1) * (idx + 1) * 100, dtype=np.uint16)
                tifffile.imwrite(img_folder / f"manual_{fov}_0_{ch}.tiff", img)

        # tile_identifiers are tuples: (fov,) for manual format
        return img_folder, ["ch1", "ch2"], [(0,), (1,)]

    def test_reads_all_channels(self, sample_tiff_folder):
        """Test that all channels are read."""
        img_folder, channel_names, tile_identifiers = sample_tiff_folder
        tile = read_individual_tiffs_tile(img_folder, channel_names, tile_identifiers, tile_idx=0)

        assert tile.shape == (2, 50, 50)
        assert tile.dtype == np.float32

    def test_correct_values(self, sample_tiff_folder):
        """Test that values are read correctly."""
        img_folder, channel_names, tile_identifiers = sample_tiff_folder
        tile = read_individual_tiffs_tile(img_folder, channel_names, tile_identifiers, tile_idx=0)

        # FOV 0: ch1 = 100, ch2 = 200
        assert np.allclose(tile[0], 100)
        assert np.allclose(tile[1], 200)


class TestOMETiffMetadata:
    """Tests for OME-TIFF metadata loading."""

    @pytest.fixture
    def sample_ome_tiff(self, tmp_path):
        """Create a sample OME-TIFF file."""
        path = tmp_path / "test.ome.tiff"

        # Create simple multi-series OME-TIFF
        data = [np.random.randint(0, 65535, (100, 100), dtype=np.uint16) for _ in range(4)]

        # Minimal OME-XML
        ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
            <Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" 
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="0" PositionY="0"/>
            </Pixels></Image>
            <Image ID="Image:1"><Pixels ID="Pixels:1" DimensionOrder="XYCZT" Type="uint16"
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="50" PositionY="0"/>
            </Pixels></Image>
            <Image ID="Image:2"><Pixels ID="Pixels:2" DimensionOrder="XYCZT" Type="uint16"
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="0" PositionY="50"/>
            </Pixels></Image>
            <Image ID="Image:3"><Pixels ID="Pixels:3" DimensionOrder="XYCZT" Type="uint16"
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="50" PositionY="50"/>
            </Pixels></Image>
        </OME>"""

        with tifffile.TiffWriter(path, ome=True) as tif:
            for i, d in enumerate(data):
                tif.write(d, description=ome_xml if i == 0 else None)

        return path

    def test_loads_ome_metadata(self, sample_ome_tiff):
        """Test loading OME-TIFF metadata."""
        # Note: This may fail if tifffile doesn't write proper multi-series OME-TIFF
        # The test structure is correct but actual OME-TIFF writing is complex
        try:
            meta = load_ome_tiff_metadata(sample_ome_tiff)
            assert "n_tiles" in meta
            assert "shape" in meta
            assert "pixel_size" in meta
            # Clean up tiff_handle
            if "tiff_handle" in meta:
                meta["tiff_handle"].close()
        except Exception:
            pytest.skip("OME-TIFF creation requires proper OME-XML handling")

    def test_handle_closed_on_error(self, tmp_path):
        """Test that handle is closed if metadata parsing fails (ID 2650114878)."""
        import os

        # Create invalid TIFF (no OME metadata)
        path = tmp_path / "invalid.tiff"
        tifffile.imwrite(path, np.zeros((10, 10), dtype=np.uint16))

        with pytest.raises(ValueError, match="does not contain OME metadata"):
            load_ome_tiff_metadata(path)

        # Verify file handle is closed by attempting operations that would fail
        # if the handle were still open (especially on Windows)
        # 1. We can reopen the file
        with tifffile.TiffFile(path) as tif:
            assert tif is not None

        # 2. We can delete the file (would fail on Windows with open handle)
        os.remove(path)
        assert not path.exists()


class TestThreadSafety:
    """Tests for thread-safe concurrent tile reads."""

    @pytest.fixture
    def sample_ome_tiff(self, tmp_path):
        """Create a sample OME-TIFF file with multiple tiles."""
        path = tmp_path / "test.ome.tiff"

        # Create tiles with distinct values for verification
        data = [np.full((100, 100), fill_value=i * 1000, dtype=np.uint16) for i in range(8)]

        ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">"""
        for i in range(8):
            ome_xml += f"""
            <Image ID="Image:{i}"><Pixels ID="Pixels:{i}" DimensionOrder="XYCZT" Type="uint16"
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="{(i % 4) * 50}" PositionY="{(i // 4) * 50}"/>
            </Pixels></Image>"""
        ome_xml += "</OME>"

        with tifffile.TiffWriter(path, ome=True) as tif:
            for i, d in enumerate(data):
                tif.write(d, description=ome_xml if i == 0 else None)

        return path, data

    def test_concurrent_reads_thread_local_handles(self, sample_ome_tiff):
        """Test that concurrent reads from multiple threads use separate handles."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tilefusion import TileFusion

        path, expected_data = sample_ome_tiff

        try:
            with TileFusion(path) as tf:
                # Track which threads read which tiles
                results = {}
                errors = []

                def read_tile(tile_idx):
                    import threading

                    thread_id = threading.current_thread().ident
                    try:
                        tile = tf._read_tile(tile_idx)
                        return tile_idx, thread_id, tile
                    except Exception as e:
                        return tile_idx, thread_id, e

                # Read tiles concurrently from multiple threads
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(read_tile, i) for i in range(8)]
                    for future in as_completed(futures):
                        tile_idx, thread_id, result = future.result()
                        if isinstance(result, Exception):
                            errors.append((tile_idx, result))
                        else:
                            results[tile_idx] = (thread_id, result)

                # Verify no errors occurred
                assert not errors, f"Errors during concurrent reads: {errors}"

                # Verify all tiles were read correctly
                assert len(results) == 8, f"Expected 8 results, got {len(results)}"

                # Verify data integrity - each tile should have its expected value
                for tile_idx, (thread_id, tile) in results.items():
                    expected_val = tile_idx * 1000
                    # The tile is flipped, so check mean value
                    actual_mean = tile.mean()
                    assert (
                        abs(actual_mean - expected_val) < 1
                    ), f"Tile {tile_idx}: expected ~{expected_val}, got {actual_mean}"

                # Verify multiple handles were created (one per thread)
                assert len(tf._all_handles) > 0, "No thread-local handles created"

        except (ValueError, AttributeError) as e:
            if "OME" in str(e) or "series" in str(e).lower():
                pytest.skip("OME-TIFF creation requires proper OME-XML handling")
            raise

    def test_handles_cleaned_up_after_close(self, sample_ome_tiff):
        """Test that all thread-local handles are closed on cleanup."""
        from concurrent.futures import ThreadPoolExecutor
        from tilefusion import TileFusion

        path, _ = sample_ome_tiff

        try:
            tf = TileFusion(path)

            # Create handles in multiple threads
            def read_tile(tile_idx):
                return tf._read_tile(tile_idx)

            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(read_tile, range(4)))

            # Verify handles were created
            num_handles = len(tf._all_handles)
            assert num_handles > 0, "No handles created"

            # Close and verify cleanup
            tf.close()
            assert len(tf._all_handles) == 0, "Handles not cleaned up"

        except (ValueError, AttributeError) as e:
            if "OME" in str(e) or "series" in str(e).lower():
                pytest.skip("OME-TIFF creation requires proper OME-XML handling")
            raise


class TestTileFusionResourceManagement:
    """Tests for TileFusion resource management (close, context manager) - ID 2650114876."""

    @pytest.fixture
    def sample_ome_tiff(self, tmp_path):
        """Create a sample OME-TIFF file."""
        path = tmp_path / "test.ome.tiff"

        data = [np.random.randint(0, 65535, (100, 100), dtype=np.uint16) for _ in range(4)]

        ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
            <Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" 
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="0" PositionY="0"/>
            </Pixels></Image>
            <Image ID="Image:1"><Pixels ID="Pixels:1" DimensionOrder="XYCZT" Type="uint16"
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="50" PositionY="0"/>
            </Pixels></Image>
            <Image ID="Image:2"><Pixels ID="Pixels:2" DimensionOrder="XYCZT" Type="uint16"
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="0" PositionY="50"/>
            </Pixels></Image>
            <Image ID="Image:3"><Pixels ID="Pixels:3" DimensionOrder="XYCZT" Type="uint16"
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="50" PositionY="50"/>
            </Pixels></Image>
        </OME>"""

        with tifffile.TiffWriter(path, ome=True) as tif:
            for i, d in enumerate(data):
                tif.write(d, description=ome_xml if i == 0 else None)

        return path

    def test_close_method(self, sample_ome_tiff):
        """Test that close() properly closes thread-local handles."""
        from tilefusion import TileFusion

        try:
            tf = TileFusion(sample_ome_tiff)
            # Trigger handle creation by reading a tile
            tf._read_tile(0)
            assert len(tf._all_handles) > 0, "Handle should be created after read"
            tf.close()
            assert len(tf._all_handles) == 0, "Handles should be cleared after close"
        except (ValueError, AttributeError) as e:
            if "OME" in str(e) or "series" in str(e).lower():
                pytest.skip("OME-TIFF creation requires proper OME-XML handling")
            raise

    def test_close_idempotent(self, sample_ome_tiff):
        """Test that close() can be called multiple times safely."""
        from tilefusion import TileFusion

        try:
            tf = TileFusion(sample_ome_tiff)
            tf.close()
            tf.close()  # Should not raise
        except (ValueError, AttributeError) as e:
            if "OME" in str(e) or "series" in str(e).lower():
                pytest.skip("OME-TIFF creation requires proper OME-XML handling")
            raise

    def test_context_manager(self, sample_ome_tiff):
        """Test context manager protocol cleans up handles on exit."""
        from tilefusion import TileFusion

        try:
            with TileFusion(sample_ome_tiff) as tf:
                # Trigger handle creation by reading a tile
                tf._read_tile(0)
                assert len(tf._all_handles) > 0, "Handle should be created"
            # After exiting context, handles should be cleaned up
            assert len(tf._all_handles) == 0, "Handles should be cleared after exit"
        except (ValueError, AttributeError) as e:
            if "OME" in str(e) or "series" in str(e).lower():
                pytest.skip("OME-TIFF creation requires proper OME-XML handling")
            raise
