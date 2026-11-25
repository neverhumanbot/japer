# Isosyll v4.0.0 — Technical & Architecture Brief

> **Download formatted 2-page PDF:** [**`isosyll-v4-cto-brief.pdf`**](isosyll-v4-cto-brief.pdf) *(publication layout with the architecture diagram, speed-vs-accuracy chart, paired head-to-head and full measured table)*

**A Bit-Exact Streaming Speech Recognizer in One Pure-Rust Binary**

## 1. Executive Technical Summary

- **One binary:** `isosyll` v4.0.0 is a ~1.6 MB Rust executable with no Python, PyTorch, ONNX Runtime or BLAS, for Linux x86-64, Linux arm64 and macOS on Apple silicon. The model is a 60 MB INT8 safetensors file plus two n-gram LMs (~207 MB download).
- **Live +LM (160 ms chunks):** 32.7× real time on one Zen 2 core (3.1% of the core), 5.54% pooled WER, 109 MB peak RSS, 0.328 J active core energy per audio-second.
- **Final (commit pass):** 5.00% WER at 16.8×; versus Zipformer INT8 at 5.25%, the paired difference is −0.26 pp with bounds −0.57 to +0.05 (includes zero).
- **Paired head-to-head:** 2.33× Zipformer INT8's throughput at 57% of its core energy and 72% of its memory, same core, same interleaved blocks.
- **Bit-exact:** byte-identical transcripts to the signed reference decoders on all 730 panel utterances on x86-64 and Apple silicon (NEON).

## 2. Architecture

- **Model:** dense 16-block streaming Conformer RNN-T with a stateless predictor (last two non-blank tokens) and a 320-wide joiner over a 500-token BPE vocabulary plus blank. 80-band log-mel at 100 Hz; a causal two-layer conv stem subsamples ×4 to 25 Hz at width 384. Blocks are macaron-style: half-step FFN, 6-head × 64 self-attention with rotary positions, causal depthwise conv (kernel 15), half-step FFN.
- **Live mode:** 160 ms chunks (4 frames), 128-frame left context, zero right context; every encoder frame depends only on audio up to the end of its chunk, so streaming inference matches training. Words are emitted continuously with no endpoint detector; a compact live n-gram LM is fused in.
- **Commit pass:** at each utterance end the buffered segment (≤ 40 s) is re-encoded in 640 ms chunks and decoded with a width-16 beam fusing a 5-gram word LM with internal log-density-ratio (LODR) subtraction. Same INT8 encoder and joiner.

## 3. Engine: Per-Row Centred INT8, Exact Integer Kernels

- Weights are quantised per output channel (`s_w = max|w| / 127`). Activations are quantised per frame row, centred on that row's range (`c = (x_max + x_min) / 2`, `s_a = (x_max − x_min) / 254`, round-half-even), so one-sided post-SiLU activations use the full INT8 range. Output: `y = s_a·s_w·Σ x_q·w_q + c·w̄ + b`, with `w̄` precomputed offline.
- Kernels: x86 AVX2 (`vpmaddubsw`), AVX-512 VNNI, ARM NEON (`sdot`/`smmla`) and a scalar reference. With `|x_q|, |w_q| ≤ 127`, pairwise products sum to at most 32,258 < 2¹⁵, so int32 accumulation equals the scalar reference bit for bit.
- Batch-invariant: each row's scale depends only on that row (verified bit-identical for batch sizes 1–32).
- Portable log-mel: unfused IEEE double arithmetic with Cody–Waite reduction and polynomial sin/cos/log, so platform `libm` differences cannot flip quantisation ties.
- 18 witness decodes on Intel AVX2, AVX-512 VNNI and Apple M3 NEON reproduce the Zen 2 transcripts on all 730 panel utterances; a forced scalar backend gives 0/730 differing utterances.

## 4. Benchmark Method

- **Workload:** 730-utterance LibriSpeech DEV panel (87.7 min; 400 dev-clean, 330 dev-other, every DEV speaker), one Rust scorer and one normaliser for every system; WER pools edits over all utterances.
- **Timing:** one process, one inference thread pinned to one physical AMD Zen 2 core alone on its L3 complex, on a quiet host; median of three cold runs, model load included.
- **Paired comparison:** Isosyll and Zipformer INT8 ran in the same interleaved blocks; speed, energy and memory ratios are medians of within-block ratios; WER differences use a paired speaker-cluster bootstrap.
- **Energy:** idle-subtracted CPU-core joules per audio-second, not wall power. Compared only for systems timed in the same blocks.
- **Selection honesty:** the panel overlaps the half of DEV used for selection, so the unselected half (2,785 utterances) is reported too: live +LM trails Zipformer by +0.81 pp (bounds +0.62 to +1.02); final −0.11 pp (bounds −0.35 to +0.11).

| Recognizer (class) | Speed (× RT) | Core used | WER | Clean / other | Peak RSS | Model | Core J / audio-s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Isosyll live +LM** (streaming, 160 ms) | **32.7** | 3.1% | **5.54%** | 3.20 / 8.88 | **109 MB** | 60 MB | **0.328\*** |
| **Isosyll final** (streaming + commit) | **16.8** | 5.9% | **5.00%** | 3.03 / 7.80 | 387 MB | 60 MB | 0.661\* |
| Zipformer INT8 (streaming) | 14.0 | 7.1% | 5.25% | 3.03 / 8.43 | 152 MB | 73 MB | 0.573\* |
| Moonshine Tiny (streaming) | 8.2 | 12.2% | 9.90% | 6.54 / 14.72 | 201 MB | 45 MB | — |
| Vosk Small (streaming) | 6.9 | 14.5% | 14.90% | 10.74 / 20.86 | 257 MB | 71 MB | — |
| Vosk Large§ (streaming) | 4.1 | 24.2% | 8.80% | 5.97 / 12.85 | 7629 MB | 2866 MB | — |
| Nemotron 0.6B, 560 ms§ (streaming) | 2.4 | 42.0% | 3.56% | 2.61 / 4.93 | 987 MB | 700 MB | — |
| Parakeet TDT v3, onnx-asr (offline) | 6.7 | 15.0% | 3.36% | 2.52 / 4.58 | 1684 MB | 670 MB | — |
| Whisper.cpp Tiny§ (streaming) | 0.9 | 109.5% | 17.46% | 14.51 / 21.68 | 226 MB | 78 MB | — |
| Whisper.cpp Base§ (streaming) | 0.4 | 239.6% | 15.28% | 12.47 / 19.31 | 339 MB | 148 MB | — |

*\* Timed in the same interleaved blocks. § Speed timed on a 73-utterance subset; WER on all 730. "Core used" is CPU-seconds per audio-second. Source: [`docs/paper/data/metrics.tex`](../paper/data/metrics.tex), [`cold_dev.csv`](../paper/data/cold_dev.csv).*

Other measured operating points on the same INT8 weights (reference decoders): live greedy 160 ms at 34.2×, 6.23% WER, 74 MB, 0.322 J/audio-s; live +LM 320 ms at 33.3×, 5.50% WER (+0.65 pp vs. Zipformer on the unselected half), 0.316 J/audio-s — 2.40× Zipformer's paired throughput at 55% of its energy.

## 5. Deployment, Footprint & Supply Chain

- **Targets:** Linux x86-64, Linux arm64, macOS on Apple silicon; AVX2, AVX-512 VNNI, NEON and scalar kernels.
- **Memory:** 109 MB peak RSS for a live stream; 387 MB with the commit pass and its 5-gram LM.
- **Signed release:** `SHA256SUMS` signed with the project SSH key; the installer checks the signature and checksums before installing.

```sh
curl -fsSL https://github.com/neverhuman/jsr/releases/latest/download/install.sh | sh
isosyll transcribe meeting.wav
```

## 6. Hard Technical Limits

> - **Exploratory, DEV only:** the v3.1.0 weights in v4.0.0 missed their pre-registered accuracy gates and ship by owner decision. No test-set results are claimed; LibriSpeech test sets were read before the benchmark freeze.
> - **In-domain read English:** LibriSpeech DEV is in-domain for both Isosyll and Zipformer; conversational, accented or noisy speech may rank differently.
> - **Live accuracy gap:** +0.81 pp behind Zipformer on the unselected half; error analysis points to acoustic discrimination of common words on hard audio, and the next lever is fresh audio and capacity.
> - **Commit-pass cost:** the final pass uses 115% of Zipformer's core energy and 387 MB.
> - **Scope of claims:** CPU-core energy only (no wall power); no latency figures.

**Read next:** [CEO Executive Brief](ceo-brief.md) · [Paper (PDF)](../paper/paper.pdf) · [Measurement Data](../paper/data/) · [Release Records](../ultra/releases/) · [Install Guide](../install.md)
