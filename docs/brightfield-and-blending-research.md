# Brightfield registration & blend bandwidth — literature-grounded recommendations

Research pass before writing any algorithm code. Goal: ground the brightfield
registration fixes (channel pick + prefilter) and the blend-bandwidth choice in
what the established microscopy/mosaic stitchers actually do.

## Primary sources

- **ASHLAR** (Muhlich, Sorger et al., *Bioinformatics* 2022) — the state-of-the-art
  stitcher for highly-multiplexed whole-slide tissue imaging.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9525007/
- **MIST** (Chalfoun et al., NIST, *Scientific Reports* 2017) — stage-model +
  global error minimization. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5504007/
- **Burt & Adelson**, *A Multiresolution Spline With Application to Image Mosaics*,
  ACM TOG 1983 — the origin of multi-band (Laplacian-pyramid) blending.
- **WSI registration review** (arXiv 2502.19123, 2025) — channel selection by
  self-information/entropy in modern WSI pipelines.
- Comparative analysis of pairwise microscopy stitching, *Sci. Reports* 2024
  (s41598-024-59626-y).

---

## 1. Brightfield registration channel pick (Track A)

**What we do now:** `_auto_pick_channel` scores each channel by intensity **std**
(contrast) over a central crop and takes the argmax. Validated on fluorescence
(RNAScope, 10× mouse brain) where it tracks registration quality.

**Why it fails on S5 brightfield:** a **saturated** (blown-out) channel has *high*
std — clipping pushes pixels to the rails and widens the histogram — while having
*no* recoverable gradient texture in the seam overlaps. So std picked the saturated
green channel (NCC ≈ 0.009) over the usable red channel (NCC ≈ 0.16).

**What the literature does:**
- ASHLAR registers on a **single reference channel** (Hoechst/nuclei in IF) and
  applies the resulting shifts to all other channels — it does *not* try to
  register every channel. *"We performed stitching and registration only on the
  reference image channel … and applied the resulting positional corrections to
  all other channels."*
- Modern WSI pipelines pick the reference channel by **self-information (entropy)**,
  i.e. the most *informative* channel, not merely the highest-contrast one.

**Recommendation (careful, won't disturb the fluorescence pick):**
keep std as the base metric but multiply by a **saturation penalty**:

```
score[c] = std(crop_c) * (1 - frac_saturated_c)
frac_saturated_c = fraction of pixels at/above the clip ceiling
                   (e.g. >= 99.5th-percentile of the dtype range, or == dtype max)
```

- For fluorescence (rarely saturated) the penalty ≈ 0 → **pick is unchanged**.
  This is the key safety property the user asked for.
- For brightfield with a blown-out channel the penalty collapses that channel's
  score, so red wins.
- Also **exclude the RGB color-composite channel** from candidacy for brightfield
  (it's a derived 3-channel view, not an independent grayscale measurement).

Do **not** switch to a raw high-frequency / spectral-flatness metric — already
tried and rejected (it rewards noise and picked the signal-poor channel; noted in
the `_auto_pick_channel` docstring).

---

## 2. Brightfield registration prefilter (Track B)

**What we do now:** `register_and_score` runs `phase_cross_correlation(...,
normalization="phase")` after histogram-matching. The `"phase"` normalization is
classic phase correlation — it **whitens the magnitude spectrum** (a strong
implicit high-pass), which is already most of the benefit.

**What ASHLAR adds:** *"ASHLAR also enhances phase correlation by pre-filtering
input images with the discrete Laplacian operator (or Laplacian-of-Gaussian — LoG —
for noisy images) to eliminate auto-correlation."* Rationale: cross-correlation of
auto-correlated signals yields spurious peaks; decorrelation "substantially improves
confidence in tile alignments." No sigma/kernel size is published.

**Why it matters more for brightfield:** brightfield tiles carry strong, smooth
low-frequency shading (illumination falloff, stain density gradients). Even with
phase whitening, that low-frequency content reduces peak sharpness. A LoG prefilter
removes it and emphasizes edges/texture — exactly the structure phase correlation
locks onto.

**Recommendation:** add an **optional LoG prefilter** ahead of phase correlation,
on by default for brightfield datasets, off for fluorescence (where phase
normalization already suffices and we don't want to perturb the validated path).
LoG sigma ≈ 1–2 px is the usual starting point; expose it if needed. This is
additive and low-risk: it only changes the *measured shift*, which is already
gated downstream by the NCC edge weight + `rotation_aware_max_shift` + stage-position
fallback (our equivalent of ASHLAR's ENCC-threshold + recorded-stage fallback).

---

## 3. Tile fusion / blend bandwidth

**What ASHLAR/MIST do:** ASHLAR composites overlaps with **linear blending** (with
other user-selectable functions) — i.e. a feather, like ours. MIST likewise uses
linear blending. Neither makes multi-band the default; multi-band is the
quality ceiling, not the baseline.

**Burt & Adelson (the gold standard for ghost-free seams):** decompose each image
into a Laplacian (band-pass) pyramid and blend each band with a transition zone
whose **width is proportional to the wavelength of that band** — low frequencies
blended over a wide region, high frequencies over a narrow one — then sum the bands.
This is precisely why multi-band avoids both visible seams (low-freq blended wide)
and ghosting/blur (high-freq blended narrow).

**For our single-band feather, the equivalent rule of thumb:**
- feather width **≥ the seam overlap depth** → the inter-tile weight crosses over
  *gradually* (each pixel dominated by its nearer tile) instead of sitting on a flat
  50/50 plateau that turns a residual misalignment into a doubled feature;
- feather width **≤ ~2× the overlap** → beyond that there's no further benefit.

This is exactly the auto-blend heuristic already in the GUI (`b = max(seam*2, 10)`,
capped to tile/2 by `make_1d_profile`). So the **current single-feather choice is
literature-consistent** for linear blending.

**Recommendation:** keep the single-band feather as the default (it matches
ASHLAR/MIST practice and our seam math). Treat **Burt-Adelson multi-band** as a
future upgrade specifically for high-contrast brightfield seams, where a residual
sub-pixel misalignment is most visible — not needed for the fluorescence path.

---

## Net

- A (channel pick): add a **saturation penalty** to the std score + drop the RGB
  composite from candidacy. Safe-by-construction for fluorescence.
- B (prefilter): add an **optional LoG prefilter** before phase correlation,
  default-on for brightfield. Additive, gated downstream.
- Blend: **no change** — single-band feather is already literature-consistent;
  multi-band is a future brightfield-only enhancement.

All three are consistent with ASHLAR (the SOTA multiplexed-WSI stitcher) and
Burt-Adelson. None require touching the validated fluorescence path.
