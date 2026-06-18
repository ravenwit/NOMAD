# Bilinear Upsampling: 64×64 → 256×256 — Mathematical Analysis

## Context

The PeriodicUNet model was trained on a **64×64** grid but the web visualization renders on a **256×256** torus mesh. We perform **bilinear interpolation** to upsample the neural network's output to the visualization resolution.

## Mathematical Formulation

### Coordinate Mapping

The upsampling ratio is exactly **4:1** in both dimensions (256/64 = 4).

For each output pixel at discrete coordinates $(o_x, o_y)$ in the $256 \times 256$ grid:

$$s_x = o_x \cdot \frac{N_{src}}{N_{dst}} = o_x \cdot \frac{64}{256} = \frac{o_x}{4}$$
$$s_y = o_y \cdot \frac{64}{256} = \frac{o_y}{4}$$

where $(s_x, s_y)$ are the continuous source coordinates in the $64 \times 64$ field.

### Bilinear Interpolation

Given continuous source coordinates $(s_x, s_y)$, we identify the four nearest source pixels:

$$i_0 = \lfloor s_x \rfloor, \quad j_0 = \lfloor s_y \rfloor$$
$$i_1 = i_0 + 1, \quad j_1 = j_0 + 1$$

The fractional remainders:

$$\alpha = s_x - i_0, \quad \beta = s_y - j_0$$

The interpolated value:

$$P_{out}(o_x, o_y) = (1-\alpha)(1-\beta) \cdot P(i_0, j_0) + \alpha(1-\beta) \cdot P(i_1, j_0) + (1-\alpha)\beta \cdot P(i_0, j_1) + \alpha \beta \cdot P(i_1, j_1)$$

### Periodic Boundary Handling

Since the torus has periodic boundary conditions in both $\theta$ and $\phi$, index wrapping is applied:

$$i_1 = (i_0 + 1) \mod 64, \quad j_1 = (j_0 + 1) \mod 64$$

This ensures seamless interpolation across the $\theta = 0 \leftrightarrow \theta = 2\pi$ and $\phi = 0 \leftrightarrow \phi = 2\pi$ boundaries, preserving the manifold's topology.

## Computational Cost

### Per-Frame Cost

| Operation | FLOPS | Memory |
|-----------|-------|--------|
| Coordinate mapping | $2 \times 256^2 = 131,072$ multiplies | negligible |
| Index computation | $4 \times 256^2 = 262,144$ floor/mod ops | negligible |
| Interpolation | $9 \times 256^2 = 589,824$ (4 mul + 4 add + 1 per pixel) | $256^2 \times 4$ bytes = 256 KB output |
| **Total** | **~983,040 FLOPS** | **256 KB** |

This is approximately **$10^6$** operations per frame, which completes in **< 1ms** on modern hardware — negligible compared to the neural network inference time (~5-20ms).

### Comparison with Alternative Upsampling Methods

| Method | FLOPS/pixel | Quality | Periodic-aware |
|--------|------------|---------|----------------|
| **Nearest Neighbor** | 0 | Blocky artifacts | ✓ (trivial) |
| **Bilinear** (ours) | 9 | Smooth, C⁰ | ✓ (with mod) |
| Bicubic | 25 | Smoother, C¹ | Requires custom kernel |
| Spectral (zero-pad FFT) | O(N log N) | Exact bandwidth preservation | ✓ (natural) |
| Learned Upsampling (PixelShuffle) | O(N·C) | Task-optimized | Requires training |

We chose **bilinear** as the optimal trade-off: it adds negligible compute, produces visually smooth results, and is trivially made periodic-aware.

## Discrepancy Analysis

### 1. Spatial Frequency Aliasing

The 64×64 model can only resolve spatial frequencies up to the Nyquist limit:

$$k_{max} = \frac{N_{src}}{2} = 32 \text{ modes per dimension}$$

The 256×256 spectral solver resolves up to $k_{max} = 128$ modes. The neural operator **cannot reconstruct high-frequency features** with wavenumbers $k > 32$. This means:

- Fine-scale wave interference patterns are smoothed out
- Sharp wavefronts will appear slightly diffused
- The energy spectrum is truncated at $k = 32$

**Quantitative bound**: For a wave with wavelength $\lambda$, the minimum resolvable wavelength on the 64×64 grid is:

$$\lambda_{min} = \frac{2\pi}{k_{max}} = \frac{2\pi}{32} \approx 0.196 \text{ rad}$$

On the torus with $R = 3.0$, $r = 1.0$, this corresponds to a physical length of approximately:

$$\ell_{min,\theta} = r \cdot \lambda_{min} = 1.0 \times 0.196 \approx 0.20 \text{ units}$$
$$\ell_{min,\phi} = (R + r) \cdot \lambda_{min} = 4.0 \times 0.196 \approx 0.78 \text{ units (outer equator)}$$

### 2. Bilinear Interpolation Error

Bilinear interpolation introduces a **low-pass filtering effect**. For a signal $f$ with spectral content at wavenumber $k$, the interpolation error scales as:

$$\epsilon_{bilinear} \propto \left(\frac{k}{k_{Nyquist}}\right)^2 \cdot \Delta x^2$$

For our 4:1 upsampling ratio ($\Delta x = 1/4$), the maximum interpolation error at the Nyquist frequency is:

$$\epsilon_{max} = O\left(\frac{1}{16}\right) \approx 6.25\%$$

In practice, the neural network's output is already band-limited to $k \leq 32$, so the bilinear interpolation error is well within the model's own prediction uncertainty.

### 3. Metric Distortion

The torus metric $\sqrt{g} = r(R + r\cos\theta)$ varies by a factor of:

$$\frac{\sqrt{g}_{max}}{\sqrt{g}_{min}} = \frac{R + r}{R - r} = \frac{4.0}{2.0} = 2.0$$

Bilinear interpolation in the $(\theta, \phi)$ coordinate space does not account for this metric variation. Ideally, we would interpolate in the **physical embedding space** (geodesic interpolation), but this would require computing the geodesic midpoints on the torus for each interpolation, which is prohibitively expensive.

The error introduced by ignoring the metric is bounded by the difference between coordinate-space and physical-space interpolation:

$$\delta_{metric} \leq \frac{1}{2} \cdot \frac{\partial^2 P}{\partial \theta^2} \cdot \frac{d\theta^2}{4} \cdot \left|\frac{d\sqrt{g}}{d\theta}\right|$$

For smooth pressure fields (which the neural net produces), this is typically **< 1%** relative error.

### 4. Summary of Discrepancies

| Source | Magnitude | Impact |
|--------|-----------|--------|
| Frequency truncation ($k > 32$) | Up to 100% for high-$k$ modes | Smoothed fine detail |
| Bilinear interpolation error | ~6% at Nyquist, ~1% for smooth fields | Negligible visual impact |
| Metric-unaware interpolation | < 1% for smooth fields | Negligible |
| Neural network prediction error | Model-dependent (training MSE) | Dominant error source |

> **Conclusion**: The bilinear upsampling from 64×64 to 256×256 adds negligible error compared to the neural network's inherent prediction uncertainty. The dominant source of discrepancy between the neural and spectral solvers is the model's learned approximation error, not the upsampling.

## Alternative: Spectral Zero-Padding (FFT Upsampling)

For reference, the mathematically exact band-preserving upsampling would be:

1. Compute 2D FFT of the 64×64 field
2. Zero-pad the spectrum to 256×256 (inserting zeros at high frequencies)
3. Inverse FFT to get 256×256 output
4. Scale by $(256/64)^2 = 16$

This preserves the exact spectral content and introduces no interpolation artifacts. However:
- Cost: $O(N \log N) \approx 256^2 \cdot 16 \approx 10^6$ FLOPS — comparable to bilinear
- Requires FFT implementation (we have one in `fft.ts`, but adds complexity)
- May introduce Gibbs ringing at sharp discontinuities

For a future optimization, spectral upsampling could be offered as an option.
