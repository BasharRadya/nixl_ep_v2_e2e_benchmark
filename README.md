# NVIDIA · EP E2E benchmark

3 deployments · 15 samples

## Contents

[Results](#results) · [Experimental setup](#experimental-setup) · [Configuration](#configuration) · [Single-domain samples](#single-domain-samples) · [Multi-domain samples](#multi-domain-samples)

## Results

### Single domain · 1 × 16

**16 GPUs · 1 NVLink domain · Prefill: 1 × EP8 · Decode: 1 × EP8**

NIXL EPv2 · DeepEPv2 · NIXL EPv1

![Output-token throughput by backend and concurrency for Single domain · 1 × 16](assets/throughput-single-domain.svg)

Aggregate output throughput across all decode GPUs.

<details>
<summary>Results table</summary>

| Concurrency | Backend | Output tokens/s ↑ | Median TTFT (s) ↓ | Median End-to-end Latency (s) ↓ |
| :---: | :---: | :---: | :---: | :---: |
| **16** | NIXL EPv2 | **766.34** | **0.278** | **21.34** |
| **16** | DeepEPv2 | **823.56** | **0.286** | **19.87** |
| **16** | NIXL EPv1 | **777.38** | **0.414** | **20.97** |
| **1,024** | NIXL EPv2 | **24,129.28** | **1.389** | **42.03** |
| **1,024** | NIXL EPv1 | **21,175.79** | **0.682** | **47.63** |
| **8,192** | NIXL EPv2 | **21,781.69** | **65.348** | **369.57** |
| **8,192** | DeepEPv2 | **21,742.80** | **70.113** | **373.05** |

</details>

### Multi-domain · 2 × 8

**16 GPUs · 2 NVLink domains · Prefill: 1 × EP8 · Decode: 1 × EP8**

Each EP group spans the domains (4+4 GPUs) via RDMA.

NIXL EPv2 · DeepEPv2 · NIXL EPv1

![Output-token throughput by backend and concurrency for Multi-domain · 2 × 8](assets/throughput-two-domains.svg)

Aggregate output throughput across all decode GPUs.

<details>
<summary>Results table</summary>

| Concurrency | Backend | Output tokens/s ↑ | Median TTFT (s) ↓ | Median End-to-end Latency (s) ↓ |
| :---: | :---: | :---: | :---: | :---: |
| **16** | NIXL EPv2 | **667.07** | **0.378** | **24.50** |
| **16** | NIXL EPv1 | **534.80** | **0.758** | **30.48** |
| **1,024** | NIXL EPv2 | **21,085.07** | **1.903** | **48.43** |
| **1,024** | DeepEPv2 | **18,219.38** | **2.681** | **55.21** |
| **1,024** | NIXL EPv1 | **13,280.00** | **1.095** | **71.15** |

</details>

### Multi-domain · 4 × 8

**32 GPUs · 4 NVLink domains · Prefill: 3 × EP8 · Decode: 1 × EP8**

Each EP group spans the domains (2+2+2+2 GPUs) via RDMA.

NIXL EPv2 · DeepEPv2 · NIXL EPv1

![Output-token throughput by backend and concurrency for Multi-domain · 4 × 8](assets/throughput-four-domains.svg)

Aggregate output throughput across all decode GPUs.

<details>
<summary>Results table</summary>

| Concurrency | Backend | Output tokens/s ↑ | Median TTFT (s) ↓ | Median End-to-end Latency (s) ↓ |
| :---: | :---: | :---: | :---: | :---: |
| **16** | NIXL EPv2 | **625.33** | **0.427** | **26.14** |
| **16** | DeepEPv2 | **570.01** | **0.524** | **28.67** |
| **16** | NIXL EPv1 | **470.86** | **0.914** | **34.64** |

</details>

## Experimental setup

<details>
<summary>Model</summary>

| Setting | Value |
| :--- | :--- |
| Checkpoint | nvidia/DeepSeek-V4-Pro-NVFP4 |
| Revision | 1449d1e641023406daf6b432361486c768aad740 |
| Precision | NVFP4 expert weights and activations (W4A4, group size 16); mixed FP8 dense layers. |

</details>

Hardware: **GB200**.

## Configuration

### GPU distribution

| Deployment | Total GPUs | Prefill GPUs | Decode GPUs | Within each domain | EP-group distribution |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Single domain · 1 × 16 | 16 | 8 | 8 | 8 prefill + 8 decode | 1 × EP8 prefill + 1 × EP8 decode; 8 GPUs per group across domains |
| Multi-domain · 2 × 8 | 16 | 8 | 8 | 4 prefill + 4 decode | 1 × EP8 prefill + 1 × EP8 decode; 4 + 4 GPUs per group across domains |
| Multi-domain · 4 × 8 | 32 | 24 | 8 | 6 prefill + 2 decode | 3 × EP8 prefill + 1 × EP8 decode; 2 + 2 + 2 + 2 GPUs per group across domains |

One serving process per GPU. Prefill and decode use separate eight-GPU EP groups. Communication uses NVLink within a domain and RDMA between domains.

### Serving configuration

<details>
<summary>Single domain · 1 × 16 · C16 · 1K/1K</summary>

| Setting | NIXL EPv2 | DeepEPv2 | NIXL EPv1 |
| :--- | :---: | :---: | :---: |
| Tensor parallelism (TP) | 1 | 1 | 1 |
| Data-parallel attention (DPA) | On · 8 ranks | On · 8 ranks | On · 8 ranks |
| Pipeline parallelism (PP) | 1 | 1 | 1 |
| Serving processes per prefill GPU | 1 | 1 | 1 |
| Serving processes per decode GPU | 1 | 1 | 1 |
| Prefill max token batch / max sequences | 8,192 / 64 | 8,192 / 64 | 512 / 64 |
| Decode max token batch / max sequences | 1,024 / 1,024 | 1,024 / 1,024 | 512 / 512 |
| Prefill graph mode | Piecewise | Piecewise | Piecewise |
| Prefill graph capture maximum (tokens) | 8,192 | 8,192 | 512 |
| Decode graph mode | Full decode only | Full decode only | Full decode only |
| Max token batch | 512 | 512 | 512 |
| Decode stream interval | 50 | 50 | 50 |
| Expert GEMM | FlashInferCuteDSLExperts | FlashInferCuteDSLExperts | FlashInferCuteDSLBatchedExperts; SwiGLU clamp10 |
| Active SM count | 26 | 26 | Not configured |
| Prefer overlap | On | On | Not applicable |
| Zero-copy · prefill / decode | On / Off | Not applicable | Not applicable |
| EP dispatch format | NVFP4 + FP8 E4M3 scales | BF16 | BF16 |
| EP combine format | BF16 | BF16 | BF16 |

</details>

<details>
<summary>Single domain · 1 × 16 · C1,024 · 1K/1K</summary>

| Setting | NIXL EPv2 | NIXL EPv1 |
| :--- | :---: | :---: |
| Tensor parallelism (TP) | 1 | 1 |
| Data-parallel attention (DPA) | On · 8 ranks | On · 8 ranks |
| Pipeline parallelism (PP) | 1 | 1 |
| Serving processes per prefill GPU | 1 | 1 |
| Serving processes per decode GPU | 1 | 1 |
| Prefill max token batch / max sequences | 8,192 / 64 | 512 / 64 |
| Decode max token batch / max sequences | 1,024 / 1,024 | 512 / 512 |
| Prefill graph mode | Piecewise | Piecewise |
| Prefill graph capture maximum (tokens) | 8,192 | 512 |
| Decode graph mode | Full decode only | Full decode only |
| Max token batch | 1,024 | 512 |
| Decode stream interval | 50 | 50 |
| Expert GEMM | FlashInferCuteDSLExperts | FlashInferCuteDSLBatchedExperts; SwiGLU clamp10 |
| Active SM count | 26 | Not configured |
| Prefer overlap | On | Not applicable |
| Zero-copy · prefill / decode | On / Off | Not applicable |
| EP dispatch format | NVFP4 + FP8 E4M3 scales | BF16 |
| EP combine format | BF16 | BF16 |

</details>

<details>
<summary>Single domain · 1 × 16 · C8,192 · 1K/1K</summary>

| Setting | NIXL EPv2 | DeepEPv2 |
| :--- | :---: | :---: |
| Tensor parallelism (TP) | 1 | 1 |
| Data-parallel attention (DPA) | On · 8 ranks | On · 8 ranks |
| Pipeline parallelism (PP) | 1 | 1 |
| Serving processes per prefill GPU | 1 | 1 |
| Serving processes per decode GPU | 1 | 1 |
| Prefill max token batch / max sequences | 8,192 / 64 | 8,192 / 64 |
| Decode max token batch / max sequences | 1,024 / 1,024 | 1,024 / 1,024 |
| Prefill graph mode | Piecewise | Piecewise |
| Prefill graph capture maximum (tokens) | 8,192 | 8,192 |
| Decode graph mode | Full decode only | Full decode only |
| Max token batch | 1,024 | 1,024 |
| Decode stream interval | 50 | 50 |
| Expert GEMM | FlashInferCuteDSLExperts | FlashInferCuteDSLExperts |
| Active SM count | 26 | 26 |
| Prefer overlap | On | On |
| Zero-copy · prefill / decode | On / On | Not applicable |
| EP dispatch format | NVFP4 + FP8 E4M3 scales | BF16 |
| EP combine format | BF16 | BF16 |

</details>

<details>
<summary>Multi-domain · 2 × 8 · C16 · 1K/1K</summary>

| Setting | NIXL EPv2 | NIXL EPv1 |
| :--- | :---: | :---: |
| Tensor parallelism (TP) | 1 | 1 |
| Data-parallel attention (DPA) | On · 8 ranks | On · 8 ranks |
| Pipeline parallelism (PP) | 1 | 1 |
| Serving processes per prefill GPU | 1 | 1 |
| Serving processes per decode GPU | 1 | 1 |
| Prefill max token batch / max sequences | 8,192 / 64 | 512 / 64 |
| Decode max token batch / max sequences | 1,024 / 1,024 | 512 / 512 |
| Prefill graph mode | Piecewise | Piecewise |
| Prefill graph capture maximum (tokens) | 8,192 | 512 |
| Decode graph mode | Full decode only | Full decode only |
| Max token batch | 1,024 | 512 |
| Decode stream interval | 50 | 50 |
| Expert GEMM | FlashInferCuteDSLExperts | FlashInferCuteDSLBatchedExperts; SwiGLU clamp10 |
| Active SM count | 26 | Not configured |
| Prefer overlap | On | Not applicable |
| Zero-copy · prefill / decode | Off / Off | Not applicable |
| EP dispatch format | NVFP4 + FP8 E4M3 scales | BF16 |
| EP combine format | BF16 | BF16 |

</details>

<details>
<summary>Multi-domain · 2 × 8 · C1,024 · 1K/1K</summary>

| Setting | NIXL EPv2 | DeepEPv2 | NIXL EPv1 |
| :--- | :---: | :---: | :---: |
| Tensor parallelism (TP) | 1 | 1 | 1 |
| Data-parallel attention (DPA) | On · 8 ranks | On · 8 ranks | On · 8 ranks |
| Pipeline parallelism (PP) | 1 | 1 | 1 |
| Serving processes per prefill GPU | 1 | 1 | 1 |
| Serving processes per decode GPU | 1 | 1 | 1 |
| Prefill max token batch / max sequences | 8,192 / 64 | 8,192 / 64 | 512 / 64 |
| Decode max token batch / max sequences | 1,024 / 1,024 | 1,024 / 1,024 | 512 / 512 |
| Prefill graph mode | Piecewise | Piecewise | Piecewise |
| Prefill graph capture maximum (tokens) | 8,192 | 8,192 | 512 |
| Decode graph mode | Full decode only | Full decode only | Full decode only |
| Max token batch | 1,024 | 1,024 | 512 |
| Decode stream interval | 50 | 50 | 50 |
| Expert GEMM | FlashInferCuteDSLExperts | FlashInferCuteDSLExperts | FlashInferCuteDSLBatchedExperts; SwiGLU clamp10 |
| Active SM count | 26 | 26 | Not configured |
| Prefer overlap | On | On | Not applicable |
| Zero-copy · prefill / decode | Off / Off | Not applicable | Not applicable |
| EP dispatch format | NVFP4 + FP8 E4M3 scales | BF16 | BF16 |
| EP combine format | BF16 | BF16 | BF16 |

</details>

<details>
<summary>Multi-domain · 4 × 8 · C16 · 1K/1K</summary>

| Setting | NIXL EPv2 | DeepEPv2 | NIXL EPv1 |
| :--- | :---: | :---: | :---: |
| Tensor parallelism (TP) | 1 | 1 | 1 |
| Data-parallel attention (DPA) | On · 8 ranks | On · 8 ranks | On · 8 ranks |
| Pipeline parallelism (PP) | 1 | 1 | 1 |
| Serving processes per prefill GPU | 1 | 1 | 1 |
| Serving processes per decode GPU | 1 | 1 | 1 |
| Prefill max token batch / max sequences | 8,192 / 64 | 8,192 / 64 | 512 / 64 |
| Decode max token batch / max sequences | 1,024 / 1,024 | 1,024 / 1,024 | 512 / 512 |
| Prefill graph mode | Piecewise | Piecewise | Piecewise |
| Prefill graph capture maximum (tokens) | 8,192 | 8,192 | 512 |
| Decode graph mode | Full decode only | Full decode only | Full decode only |
| Max token batch | 512 | 512 | 512 |
| Decode stream interval | 50 | 50 | 50 |
| Expert GEMM | FlashInferCuteDSLExperts | FlashInferCuteDSLExperts | FlashInferCuteDSLBatchedExperts; SwiGLU clamp10 |
| Active SM count | 26 | 26 | Not configured |
| Prefer overlap | On | On | Not applicable |
| Zero-copy · prefill / decode | Off / Off | Not applicable | Not applicable |
| EP dispatch format | NVFP4 + FP8 E4M3 scales | BF16 | BF16 |
| EP combine format | BF16 | BF16 | BF16 |

</details>

## Single-domain samples

<details>
<summary>Single domain · 1 × 16 · C16 · 1K/1K 3 backends</summary>

#### NIXL EPv2

| Setting | Value |
| :--- | :--- |
| Output throughput | 766.34 tokens/s · all decode GPUs |
| Median TTFT | 0.278 seconds |
| Median End-to-end Latency | 21.34 seconds |

Requests: 464 · Errors: 0 · Duration: 620.00 s, including drain.

**EP wire format, prefill and decode:** dispatch NVFP4 + FP8 E4M3 scales; combine BF16. Packed FP4 activations include one E4M3 scale per 16 values. The wire format is the same with zero-copy On or Off.

#### DeepEPv2

| Setting | Value |
| :--- | :--- |
| Output throughput | 823.56 tokens/s · all decode GPUs |
| Median TTFT | 0.286 seconds |
| Median End-to-end Latency | 19.87 seconds |

Requests: 496 · Errors: 0 · Duration: 616.72 s, including drain.

**EP wire format, prefill and decode:** dispatch BF16; combine BF16. Received BF16 activations are quantized locally to NVFP4 for expert GEMM.

#### NIXL EPv1

| Setting | Value |
| :--- | :--- |
| Output throughput | 777.38 tokens/s · all decode GPUs |
| Median TTFT | 0.414 seconds |
| Median End-to-end Latency | 20.97 seconds |

Requests: 464 · Errors: 0 · Duration: 611.20 s, including drain.

**EP wire format, prefill and decode:** dispatch BF16; combine BF16. Received BF16 activations are quantized locally to NVFP4 for expert GEMM.

16 GPUs · 1 NVLink domain · Prefill: 1 × EP8 · Decode: 1 × EP8

| Setting | NIXL EPv2 | DeepEPv2 | NIXL EPv1 |
| :--- | :---: | :---: | :---: |
| Tensor parallelism (TP) | 1 | 1 | 1 |
| Data-parallel attention (DPA) | On · 8 ranks | On · 8 ranks | On · 8 ranks |
| Pipeline parallelism (PP) | 1 | 1 | 1 |
| Serving processes per prefill GPU | 1 | 1 | 1 |
| Serving processes per decode GPU | 1 | 1 | 1 |
| Prefill max token batch / max sequences | 8,192 / 64 | 8,192 / 64 | 512 / 64 |
| Decode max token batch / max sequences | 1,024 / 1,024 | 1,024 / 1,024 | 512 / 512 |
| Prefill graph mode | Piecewise | Piecewise | Piecewise |
| Prefill graph capture maximum (tokens) | 8,192 | 8,192 | 512 |
| Decode graph mode | Full decode only | Full decode only | Full decode only |
| Max token batch | 512 | 512 | 512 |
| Decode stream interval | 50 | 50 | 50 |
| Expert GEMM | FlashInferCuteDSLExperts | FlashInferCuteDSLExperts | FlashInferCuteDSLBatchedExperts; SwiGLU clamp10 |
| Active SM count | 26 | 26 | Not configured |
| Prefer overlap | On | On | Not applicable |
| Zero-copy · prefill / decode | On / Off | Not applicable | Not applicable |
| EP dispatch format | NVFP4 + FP8 E4M3 scales | BF16 | BF16 |
| EP combine format | BF16 | BF16 | BF16 |

<details>
<summary>GPU distribution and EP groups</summary>

**16 GPUs · 1 NVLink domain · Prefill: 1 × EP8 · Decode: 1 × EP8**

</details>

</details>

<details>
<summary>Single domain · 1 × 16 · C1,024 · 1K/1K 2 backends</summary>

#### NIXL EPv2

| Setting | Value |
| :--- | :--- |
| Output throughput | 24,129.28 tokens/s · all decode GPUs |
| Median TTFT | 1.389 seconds |
| Median End-to-end Latency | 42.03 seconds |

Requests: 15,093 · Errors: 0 · Duration: 640.51 s, including drain.

**EP wire format, prefill and decode:** dispatch NVFP4 + FP8 E4M3 scales; combine BF16. Packed FP4 activations include one E4M3 scale per 16 values. The wire format is the same with zero-copy On or Off.

#### NIXL EPv1

| Setting | Value |
| :--- | :--- |
| Output throughput | 21,175.79 tokens/s · all decode GPUs |
| Median TTFT | 0.682 seconds |
| Median End-to-end Latency | 47.63 seconds |

Requests: 13,312 · Errors: 0 · Duration: 643.73 s, including drain.

**EP wire format, prefill and decode:** dispatch BF16; combine BF16. Received BF16 activations are quantized locally to NVFP4 for expert GEMM.

16 GPUs · 1 NVLink domain · Prefill: 1 × EP8 · Decode: 1 × EP8

| Setting | NIXL EPv2 | NIXL EPv1 |
| :--- | :---: | :---: |
| Tensor parallelism (TP) | 1 | 1 |
| Data-parallel attention (DPA) | On · 8 ranks | On · 8 ranks |
| Pipeline parallelism (PP) | 1 | 1 |
| Serving processes per prefill GPU | 1 | 1 |
| Serving processes per decode GPU | 1 | 1 |
| Prefill max token batch / max sequences | 8,192 / 64 | 512 / 64 |
| Decode max token batch / max sequences | 1,024 / 1,024 | 512 / 512 |
| Prefill graph mode | Piecewise | Piecewise |
| Prefill graph capture maximum (tokens) | 8,192 | 512 |
| Decode graph mode | Full decode only | Full decode only |
| Max token batch | 1,024 | 512 |
| Decode stream interval | 50 | 50 |
| Expert GEMM | FlashInferCuteDSLExperts | FlashInferCuteDSLBatchedExperts; SwiGLU clamp10 |
| Active SM count | 26 | Not configured |
| Prefer overlap | On | Not applicable |
| Zero-copy · prefill / decode | On / Off | Not applicable |
| EP dispatch format | NVFP4 + FP8 E4M3 scales | BF16 |
| EP combine format | BF16 | BF16 |

<details>
<summary>GPU distribution and EP groups</summary>

**16 GPUs · 1 NVLink domain · Prefill: 1 × EP8 · Decode: 1 × EP8**

</details>

</details>

<details>
<summary>Single domain · 1 × 16 · C8,192 · 1K/1K 2 backends</summary>

#### NIXL EPv2

| Setting | Value |
| :--- | :--- |
| Output throughput | 21,781.69 tokens/s · all decode GPUs |
| Median TTFT | 65.348 seconds |
| Median End-to-end Latency | 369.57 seconds |

Requests: 16,574 · Errors: 0 · Duration: 779.17 s, including drain.

**EP wire format, prefill and decode:** dispatch NVFP4 + FP8 E4M3 scales; combine BF16. Packed FP4 activations include one E4M3 scale per 16 values. The wire format is the same with zero-copy On or Off.

#### DeepEPv2

| Setting | Value |
| :--- | :--- |
| Output throughput | 21,742.80 tokens/s · all decode GPUs |
| Median TTFT | 70.113 seconds |
| Median End-to-end Latency | 373.05 seconds |

Requests: 16,749 · Errors: 0 · Duration: 788.81 s, including drain.

**EP wire format, prefill and decode:** dispatch BF16; combine BF16. Received BF16 activations are quantized locally to NVFP4 for expert GEMM.

16 GPUs · 1 NVLink domain · Prefill: 1 × EP8 · Decode: 1 × EP8

| Setting | NIXL EPv2 | DeepEPv2 |
| :--- | :---: | :---: |
| Tensor parallelism (TP) | 1 | 1 |
| Data-parallel attention (DPA) | On · 8 ranks | On · 8 ranks |
| Pipeline parallelism (PP) | 1 | 1 |
| Serving processes per prefill GPU | 1 | 1 |
| Serving processes per decode GPU | 1 | 1 |
| Prefill max token batch / max sequences | 8,192 / 64 | 8,192 / 64 |
| Decode max token batch / max sequences | 1,024 / 1,024 | 1,024 / 1,024 |
| Prefill graph mode | Piecewise | Piecewise |
| Prefill graph capture maximum (tokens) | 8,192 | 8,192 |
| Decode graph mode | Full decode only | Full decode only |
| Max token batch | 1,024 | 1,024 |
| Decode stream interval | 50 | 50 |
| Expert GEMM | FlashInferCuteDSLExperts | FlashInferCuteDSLExperts |
| Active SM count | 26 | 26 |
| Prefer overlap | On | On |
| Zero-copy · prefill / decode | On / On | Not applicable |
| EP dispatch format | NVFP4 + FP8 E4M3 scales | BF16 |
| EP combine format | BF16 | BF16 |

<details>
<summary>GPU distribution and EP groups</summary>

**16 GPUs · 1 NVLink domain · Prefill: 1 × EP8 · Decode: 1 × EP8**

</details>

</details>

## Multi-domain samples

<details>
<summary>Multi-domain · 2 × 8 · C16 · 1K/1K 2 backends</summary>

#### NIXL EPv2

| Setting | Value |
| :--- | :--- |
| Output throughput | 667.07 tokens/s · all decode GPUs |
| Median TTFT | 0.378 seconds |
| Median End-to-end Latency | 24.50 seconds |

Requests: 400 · Errors: 0 · Duration: 614.03 s, including drain.

**EP wire format, prefill and decode:** dispatch NVFP4 + FP8 E4M3 scales; combine BF16. Packed FP4 activations include one E4M3 scale per 16 values. The wire format is the same with zero-copy On or Off.

#### NIXL EPv1

| Setting | Value |
| :--- | :--- |
| Output throughput | 534.80 tokens/s · all decode GPUs |
| Median TTFT | 0.758 seconds |
| Median End-to-end Latency | 30.48 seconds |

Requests: 320 · Errors: 0 · Duration: 612.71 s, including drain.

**EP wire format, prefill and decode:** dispatch BF16; combine BF16. Received BF16 activations are quantized locally to NVFP4 for expert GEMM.

16 GPUs · 2 NVLink domains · Prefill: 1 × EP8 · Decode: 1 × EP8

| Setting | NIXL EPv2 | NIXL EPv1 |
| :--- | :---: | :---: |
| Tensor parallelism (TP) | 1 | 1 |
| Data-parallel attention (DPA) | On · 8 ranks | On · 8 ranks |
| Pipeline parallelism (PP) | 1 | 1 |
| Serving processes per prefill GPU | 1 | 1 |
| Serving processes per decode GPU | 1 | 1 |
| Prefill max token batch / max sequences | 8,192 / 64 | 512 / 64 |
| Decode max token batch / max sequences | 1,024 / 1,024 | 512 / 512 |
| Prefill graph mode | Piecewise | Piecewise |
| Prefill graph capture maximum (tokens) | 8,192 | 512 |
| Decode graph mode | Full decode only | Full decode only |
| Max token batch | 1,024 | 512 |
| Decode stream interval | 50 | 50 |
| Expert GEMM | FlashInferCuteDSLExperts | FlashInferCuteDSLBatchedExperts; SwiGLU clamp10 |
| Active SM count | 26 | Not configured |
| Prefer overlap | On | Not applicable |
| Zero-copy · prefill / decode | Off / Off | Not applicable |
| EP dispatch format | NVFP4 + FP8 E4M3 scales | BF16 |
| EP combine format | BF16 | BF16 |

<details>
<summary>GPU distribution and EP groups</summary>

**16 GPUs · 2 NVLink domains · Prefill: 1 × EP8 · Decode: 1 × EP8**

Each EP group spans the domains (4+4 GPUs) via RDMA.

</details>

</details>

<details>
<summary>Multi-domain · 2 × 8 · C1,024 · 1K/1K 3 backends</summary>

#### NIXL EPv2

| Setting | Value |
| :--- | :--- |
| Output throughput | 21,085.07 tokens/s · all decode GPUs |
| Median TTFT | 1.903 seconds |
| Median End-to-end Latency | 48.43 seconds |

Requests: 13,312 · Errors: 0 · Duration: 646.49 s, including drain.

**EP wire format, prefill and decode:** dispatch NVFP4 + FP8 E4M3 scales; combine BF16. Packed FP4 activations include one E4M3 scale per 16 values. The wire format is the same with zero-copy On or Off.

<details>
<summary>Recorded GPU-memory activity</summary>

Memory-controller activity measures time active. Values are means of the eight per-GPU time averages for each role.

| Role | Mean activity | Highest recorded peak | Minimum window coverage |
| :--- | :---: | :---: | :---: |
| Prefill | 13.13% | 31.00% | 98.99% |
| Decode | 46.80% | 48.00% | 98.99% |

KV-cache pool occupancy peaked at 27.25% across the exported engine summaries. This is the allocated KV pool, rather than total device-memory occupancy. Memory-controller activity is not a bandwidth measurement in GB/s.

</details>

#### DeepEPv2

| Setting | Value |
| :--- | :--- |
| Output throughput | 18,219.38 tokens/s · all decode GPUs |
| Median TTFT | 2.681 seconds |
| Median End-to-end Latency | 55.21 seconds |

Requests: 11,465 · Errors: 0 · Duration: 644.37 s, including drain.

**EP wire format, prefill and decode:** dispatch BF16; combine BF16. Received BF16 activations are quantized locally to NVFP4 for expert GEMM.

<details>
<summary>Recorded GPU-memory activity</summary>

Memory-controller activity measures time active. Values are means of the eight per-GPU time averages for each role.

| Role | Mean activity | Highest recorded peak | Minimum window coverage |
| :--- | :---: | :---: | :---: |
| Prefill | 11.52% | 26.00% | 99.32% |
| Decode | 41.91% | 43.00% | 99.32% |

KV-cache pool occupancy peaked at 32.01% across the exported engine summaries. This is the allocated KV pool, rather than total device-memory occupancy. Memory-controller activity is not a bandwidth measurement in GB/s.

</details>

<details>
<summary>Startup memory components · sampled rank 0</summary>

Recorded device capacity: 184.31 GiB per sampled GPU. The startup profiling components below are separate observations; their sum is not a simultaneous total or a measured runtime HBM peak.

| Component | Prefill (GiB) | Decode (GiB) |
| :--- | :---: | :---: |
| Model loading | 128.65 | 128.63 |
| Weights and non-PyTorch consumption | 134.71 | 131.48 |
| Profiled peak activations | 16.37 | 2.83 |
| Graph capture pool | 12.41 | 1.82 |
| Current KV allocation | 14.79 | 31.56 |

</details>

#### NIXL EPv1

| Setting | Value |
| :--- | :--- |
| Output throughput | 13,280.00 tokens/s · all decode GPUs |
| Median TTFT | 1.095 seconds |
| Median End-to-end Latency | 71.15 seconds |

Requests: 8,781 · Errors: 0 · Duration: 677.09 s, including drain.

**EP wire format, prefill and decode:** dispatch BF16; combine BF16. Received BF16 activations are quantized locally to NVFP4 for expert GEMM.

<details>
<summary>Recorded GPU-memory activity</summary>

Memory-controller activity measures time active. Values are means of the eight per-GPU time averages for each role.

| Role | Mean activity | Highest recorded peak | Minimum window coverage |
| :--- | :---: | :---: | :---: |
| Prefill | 11.85% | 17.50% | 98.95% |
| Decode | 32.75% | 42.50% | 98.95% |

KV-cache pool occupancy peaked at 53.28% across the exported engine summaries. This is the allocated KV pool, rather than total device-memory occupancy. Memory-controller activity is not a bandwidth measurement in GB/s.

</details>

16 GPUs · 2 NVLink domains · Prefill: 1 × EP8 · Decode: 1 × EP8

| Setting | NIXL EPv2 | DeepEPv2 | NIXL EPv1 |
| :--- | :---: | :---: | :---: |
| Tensor parallelism (TP) | 1 | 1 | 1 |
| Data-parallel attention (DPA) | On · 8 ranks | On · 8 ranks | On · 8 ranks |
| Pipeline parallelism (PP) | 1 | 1 | 1 |
| Serving processes per prefill GPU | 1 | 1 | 1 |
| Serving processes per decode GPU | 1 | 1 | 1 |
| Prefill max token batch / max sequences | 8,192 / 64 | 8,192 / 64 | 512 / 64 |
| Decode max token batch / max sequences | 1,024 / 1,024 | 1,024 / 1,024 | 512 / 512 |
| Prefill graph mode | Piecewise | Piecewise | Piecewise |
| Prefill graph capture maximum (tokens) | 8,192 | 8,192 | 512 |
| Decode graph mode | Full decode only | Full decode only | Full decode only |
| Max token batch | 1,024 | 1,024 | 512 |
| Decode stream interval | 50 | 50 | 50 |
| Expert GEMM | FlashInferCuteDSLExperts | FlashInferCuteDSLExperts | FlashInferCuteDSLBatchedExperts; SwiGLU clamp10 |
| Active SM count | 26 | 26 | Not configured |
| Prefer overlap | On | On | Not applicable |
| Zero-copy · prefill / decode | Off / Off | Not applicable | Not applicable |
| EP dispatch format | NVFP4 + FP8 E4M3 scales | BF16 | BF16 |
| EP combine format | BF16 | BF16 | BF16 |

<details>
<summary>GPU distribution and EP groups</summary>

**16 GPUs · 2 NVLink domains · Prefill: 1 × EP8 · Decode: 1 × EP8**

Each EP group spans the domains (4+4 GPUs) via RDMA.

</details>

</details>

<details>
<summary>Multi-domain · 4 × 8 · C16 · 1K/1K 3 backends</summary>

#### NIXL EPv2

| Setting | Value |
| :--- | :--- |
| Output throughput | 625.33 tokens/s · all decode GPUs |
| Median TTFT | 0.427 seconds |
| Median End-to-end Latency | 26.14 seconds |

Requests: 368 · Errors: 0 · Duration: 602.61 s, including drain.

**EP wire format, prefill and decode:** dispatch NVFP4 + FP8 E4M3 scales; combine BF16. Packed FP4 activations include one E4M3 scale per 16 values. The wire format is the same with zero-copy On or Off.

#### DeepEPv2

| Setting | Value |
| :--- | :--- |
| Output throughput | 570.01 tokens/s · all decode GPUs |
| Median TTFT | 0.524 seconds |
| Median End-to-end Latency | 28.67 seconds |

Requests: 336 · Errors: 0 · Duration: 603.60 s, including drain.

**EP wire format, prefill and decode:** dispatch BF16; combine BF16. Received BF16 activations are quantized locally to NVFP4 for expert GEMM.

#### NIXL EPv1

| Setting | Value |
| :--- | :--- |
| Output throughput | 470.86 tokens/s · all decode GPUs |
| Median TTFT | 0.914 seconds |
| Median End-to-end Latency | 34.64 seconds |

Requests: 288 · Errors: 0 · Duration: 626.32 s, including drain.

**EP wire format, prefill and decode:** dispatch BF16; combine BF16. Received BF16 activations are quantized locally to NVFP4 for expert GEMM.

32 GPUs · 4 NVLink domains · Prefill: 3 × EP8 · Decode: 1 × EP8

| Setting | NIXL EPv2 | DeepEPv2 | NIXL EPv1 |
| :--- | :---: | :---: | :---: |
| Tensor parallelism (TP) | 1 | 1 | 1 |
| Data-parallel attention (DPA) | On · 8 ranks | On · 8 ranks | On · 8 ranks |
| Pipeline parallelism (PP) | 1 | 1 | 1 |
| Serving processes per prefill GPU | 1 | 1 | 1 |
| Serving processes per decode GPU | 1 | 1 | 1 |
| Prefill max token batch / max sequences | 8,192 / 64 | 8,192 / 64 | 512 / 64 |
| Decode max token batch / max sequences | 1,024 / 1,024 | 1,024 / 1,024 | 512 / 512 |
| Prefill graph mode | Piecewise | Piecewise | Piecewise |
| Prefill graph capture maximum (tokens) | 8,192 | 8,192 | 512 |
| Decode graph mode | Full decode only | Full decode only | Full decode only |
| Max token batch | 512 | 512 | 512 |
| Decode stream interval | 50 | 50 | 50 |
| Expert GEMM | FlashInferCuteDSLExperts | FlashInferCuteDSLExperts | FlashInferCuteDSLBatchedExperts; SwiGLU clamp10 |
| Active SM count | 26 | 26 | Not configured |
| Prefer overlap | On | On | Not applicable |
| Zero-copy · prefill / decode | Off / Off | Not applicable | Not applicable |
| EP dispatch format | NVFP4 + FP8 E4M3 scales | BF16 | BF16 |
| EP combine format | BF16 | BF16 | BF16 |

<details>
<summary>GPU distribution and EP groups</summary>

**32 GPUs · 4 NVLink domains · Prefill: 3 × EP8 · Decode: 1 × EP8**

Each EP group spans the domains (2+2+2+2 GPUs) via RDMA.

</details>

</details>

Generated 2026-10-21 18:47 UTC
