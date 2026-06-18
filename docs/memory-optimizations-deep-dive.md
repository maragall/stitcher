# Stitcher Memory: A History of Three Choices

The three commits that lowered the stitcher's memory, in fusion and in display: the code
before each, the mechanism that was wrong, the alternatives, and why I chose what I chose.
These are space-complexity fixes. In each case memory scaled with something it should not
have, the machine's RAM or the dataset size, and the fix makes it a fixed constant.

## How chunked fusion works

The microscope acquires a grid of overlapping **FOVs**, one camera image each, about
2084 x 2084 px on Squid. Each FOV is a separate array on disk with a known position, in
pixels, on the output **plane**: the coordinate space of the final mosaic for one z-level
and one timepoint, sized `padded_shape = (pad_Y, pad_X)`. The plane is not a stored image.
It does not exist yet; fusion produces it. Fusion runs one plane at a time, with z and
timepoint as the outer loops.

Fusion never loads the whole plane. It lays a regular, non-overlapping grid of **blocks**
over the plane's coordinate space, `block_size` pixels per side, and produces one block at
a time:

```python
# each FOV's pixel rectangle on the plane, from its stage position
tile_bounds = [(oy, oy + self.Y, ox, ox + self.X) for ...]

for block_y in range(0, pad_Y, block_size):       # grid of blocks tiling the plane
    for block_x in range(0, pad_X, block_size):
        # FOVs whose rectangle intersects this block
        overlapping = [i for i, (ty0, ty1, tx0, tx1) in enumerate(tile_bounds)
                       if ty1 > block_y and ty0 < by_end and tx1 > block_x and tx0 < bx_end]
        if not overlapping:
            continue
        # read each overlapping FOV, blend its part into the block, normalize, write block
```

Blocks do not overlap each other; FOVs do, and that overlap is the seam, which fusion
blends as a weighted average. So a block usually holds pieces of several FOVs, and one FOV
usually spans several blocks. Block and FOV are different objects: the FOV is a fixed-size
input at an arbitrary, overlapping position; the block is a cell of a regular grid laid
over the output. Only one block, plus the FOVs that touch it, is in memory at a time, which
is what bounds memory no matter how large the plane is.

The block size is tied to the output storage. The plane is written as a Zarr v3 array in
**chunks** of `chunk_y` x `chunk_x` = 1024 x 1024 px. `block_size = chunk_y * 2 = 2048`, so
one block is a 2 x 2 group of chunks and writes aligned to the storage.

## Principle

Memory should be fixed by the design of the function: a constant, independent of the
machine's RAM and of the dataset size. A runtime cap, constraining memory while the
function runs, is a feature that hides what is fundamentally wrong with the function
instead of fixing it. It needs live monitoring and a stall-or-abort path, and the function
still grows underneath. The fix is to make the function not grow.

---

## 1. The fusion buffers were reallocated on every block (`a6840e5`) `[attach commit link]`

In the block loop above, each block keeps two accumulators its own size: `fused` (sum of
pixel x weight) and `weight` (sum of weight). After all overlapping FOVs are added,
`fused / weight` is the blended block.

```python
bytes_per_pixel = 4 * 2 * self.channels        # used to size the block from RAM (see §2)
...
fused_block = np.zeros((C, bh, bw), dtype=np.float32)   # new arrays, every block
weight_sum  = np.zeros_like(fused_block)
...                                            # accumulation loop over overlapping FOVs
mask = weight_sum > 0
fused_block[mask] /= weight_sum[mask]
...
del fused_block, weight_sum
```

`C = self.channels` (fluorescence channels); `bh, bw` are block height and width. Arrays
are `(channel, y, x)`; there is no z because the function fuses one plane.

Three problems:

1. **Reallocation per block.** `np.zeros` runs inside the loop and `del` frees it each
   iteration, so every block allocates and frees both buffers. Repeated allocate-then-free
   of the same shapes is churn: GC pressure and a sawtooth footprint, for buffers that
   could be allocated once.

2. **The divide copied.** `fused_block[mask] /= weight_sum[mask]` is a boolean-index
   divide: it extracts the masked elements into two new arrays, divides those, and writes
   back. Those temporaries are nearly block-sized, allocated on top of the buffers already
   in memory, so the block's footprint roughly doubles during normalization. (The mask
   skips pixels no FOV covered, where weight is 0 and the divide is undefined.)

3. **The byte estimate was low.** `4 * 2 * channels` is float32 (4) times two accumulators
   (2) times channels, so 8 bytes per pixel per channel. The real working set is fused (4)
   + weight (4) + mask (1) + the uint16 copy made to write (2), about 12. The RAM-derived
   block was a third larger than the budget assumed.

Alternatives for the divide: divide everything and clean up the zero-weight infinities
after, which is an extra pass; or a numba per-pixel kernel, which is what the whole-plane
path uses but leaves vectorized numpy.

Chosen: allocate the buffers once, view-and-zero them per block, and divide in place.

```python
fused_buf  = np.zeros((C, max_bh, max_bw), dtype=np.float32)   # allocated once
weight_buf = np.zeros((C, max_bh, max_bw), dtype=np.float32)
...
fused_block = fused_buf[:, :bh, :bw]; fused_block[...] = 0.0
weight_sum  = weight_buf[:, :bh, :bw]; weight_sum[...]  = 0.0
...
np.divide(fused_block, weight_sum, out=fused_block, where=mask)
```

`out=fused_block` writes the result into the existing array; `where=mask` computes only at
the covered pixels. Same arithmetic, no per-block allocation, no temporaries. Output is
byte-identical, pinned by `tests/test_fuse_equivalence.py`. Peak RSS 3971 to 2527 MB
(down 36%), about 20% faster.

---

## 2. The block size was taken from free RAM, with a whole-plane switch (`62f267a`) `[attach commit link]`

Same function and original author (`6958207`). The block size was computed from free RAM,
clamped, and past a threshold the function abandoned the block grid and loaded the whole
plane.

```python
available_ram = psutil.virtual_memory().available
usable_ram = int(available_ram * ram_fraction)              # ram_fraction default 0.4
block_size = int(np.sqrt(usable_ram // (12 * self.channels)))   # grows with free RAM
block_size = min(block_size, 10240)                         # ceiling: 10 chunks (10 * 1024)
...
if block_size >= max(pad_Y, pad_X):                         # one block as big as the plane
    return self._fuse_tiles_full_plane(...)                 # drop the grid, load whole plane
```

Core failure: **more RAM, bigger block.** `block_size` grows with free RAM, the opposite
of safe.

Two details make it worse. The 10240 ceiling (10 x 1024) is arbitrary; at it, the two
accumulators alone are about 3.4 GB at 4 channels. And the switch does not just enlarge the
block, it calls `_fuse_tiles_full_plane`, which allocates the entire plane
(`np.zeros((1, channels, 1, pad_Y, pad_X), uint16)` plus full-plane float32 accumulators),
with no bound from the mosaic size. On a 30 GB machine the formula resolved into a multi-GB
block or crossed into the whole-plane path and filled memory. That is the crash.

Alternatives:

- Lower the 0.4 fraction or the 10240 ceiling: still grows with RAM, still has the switch.
  Tunes the symptom.
- Enforce the ceiling at runtime: check each block against a budget while running and
  stall or shrink on exceed. Runtime machinery for a quantity that never needed to grow,
  since fusion holds only one block at a time.

Chosen: fix the block to the storage layout. Fusion writes one block at a time, so it
never needs more than one block in memory.

```python
block_size = self.chunk_y * 2   # 2048, one block = a 2 x 2 group of chunks
```

This deleted psutil, `ram_fraction`, the ceiling, and the switch (11 lines removed). The
block buffers are now about 167 MB at 4 channels on every machine, and no path loads the
whole plane.

The whole-plane path is not faster, it only does less I/O. A fixed block re-reads any FOV
that straddles a block boundary, but the compute is identical. Measured on `test_10x`,
same machine: whole-plane 60.3 s, fixed block 60.9 s, within 1%. The re-reads hit the warm
page cache and each FOV is about 9 MB, so they do not move the wall-clock. There is no
speed argument for holding the plane in RAM. For a much larger mosaic the re-reads grow but
stay bounded, and the answer then is a larger fixed block, not the plane.

Output is byte-identical (`test_fuse_equivalence`). Peak RSS fell to 1447 MB, and the peak
moved off the Fuse stage onto Write.

---

## 3. The viewer read whole volumes into RAM (`e432cbf`, `b047d0b`) `[attach commit link]`

Display side (`gui/app.py`), separate from fusion. "Open in Napari" filled memory even
though fusion stayed bounded.

Per channel, the viewer read the entire volume eagerly and handed napari a plain numpy
array. Hongquan Li (`832d1d6`, 2025-12-27):

```python
data = store[:, c, :, :, :].read().result()   # whole volume, all z and t, this channel
data = np.asarray(data)
```

napari only displays the current slice at the current zoom; it never needs the whole
volume. My earlier `7c32ebe` (2026-03-30) cut this to a 4x downsampled level but kept the
eager read, so it still scaled with the data.

The fix is to hand napari the multiscale pyramid as lazy arrays, so it reads only the chunk
and level it renders. The correctness bar for lazy code: the object must report shape and
dtype without reading, and read only the requested slice on `__getitem__`. If slicing reads
the whole array first, it is eager behind a lazy interface.

First attempt (`e432cbf`): lazy via ome-zarr. It crashed with `'NoneType' object has no
attribute 'exists'`. `parse_url` returns None when it cannot open a store, and that None
was passed into `Reader`. Root cause: ome-zarr's Zarr v3 support is version-dependent and
could not open our store on some machines.

Alternatives:

- Eager, downsample harder (my March approach): still reads a whole volume.
- Lazy via ome-zarr: the None crash, version-fragile across machines.

Chosen (`b047d0b`): read each level with tensorstore, the library that writes our Zarr v3
output, so it always opens it. Wrap each level in a dask array. The adapter:

```python
class _TSArray:
    def __init__(self, store):
        self._store = store
        self.shape = tuple(store.shape)          # reported without reading
        self.dtype = store.dtype.numpy_dtype
        self.ndim  = len(self.shape)
    def __getitem__(self, idx):
        return np.asarray(self._store[idx].read().result())   # reads only idx
...
store = ts.open({"driver": "zarr3",
                 "kvstore": {"driver": "file", "path": str(image_path)}}).result()
levels.append(da.from_array(_TSArray(store), chunks=...))
```

The path is napari to dask to `_TSArray` to tensorstore, each reading only the requested
chunk. `__getitem__` forwards `idx` straight to the store, so it is genuinely lazy. Display
memory is flat regardless of dataset size, and the viewer no longer depends on ome-zarr.

---

## Still RAM-shaped, and one duplication

Three things this work did not touch:

1. **Registration** still sizes its work from free RAM, the same pattern as §2. Next target.

2. **Direct-placement mode** (`_fuse_tiles_direct_plane`, `core.py:943`) has its own RAM switch,
   reached when fusion runs with blending off (`mode="direct"`, set when the GUI Blend checkbox
   is unchecked):
   ```python
   available_ram = psutil.virtual_memory().available
   output_bytes  = pad_Y * pad_X * self.channels * 2          # uint16 plane
   use_memory    = output_bytes < 0.45 * available_ram
   if use_memory:
       output = np.zeros((1, self.channels, 1, pad_Y, pad_X), dtype=np.uint16)   # whole plane
   ```
   It is the same machine-dependent pattern as §2: the path taken depends on free RAM, and the
   in-memory branch allocates the whole plane. It is much less dangerous than the §2 switch was,
   because it is uint16 with no weight buffer (half the bytes, one array) and it falls back to a
   tile-by-tile streaming write when the plane does not fit, so it cannot exceed 45% of RAM or
   reproduce the crash. The fix is the §2 fix: delete the in-memory branch and the psutil check
   and always stream (the streaming code already exists as the `else` branch at `core.py:998`),
   making direct mode bounded and machine-independent.

3. **Two implementations of the same blend.** The whole-plane path (`_fuse_tiles_full_plane`)
   accumulates and normalizes with the numba kernels (`accumulate_tile_shard` / `normalize_shard`);
   the chunked path does the same arithmetic inline in numpy. They agree (the equivalence test
   passes), but no caller sets `chunked=False`, so `_fuse_tiles_full_plane` only runs as the
   reference inside `test_fuse_equivalence`. That means the test's ground truth executes different
   code from production, a correctness risk to watch, not a runtime memory trigger.
