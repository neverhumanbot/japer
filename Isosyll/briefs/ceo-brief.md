# Isosyll v4.0.0 — Executive Brief

> **Download formatted 2-page PDF:** [**`isosyll-v4-ceo-brief.pdf`**](isosyll-v4-ceo-brief.pdf) *(publication layout with the speed-vs-accuracy chart and the same-core head-to-head)*

**Always Listening, Barely Computing: Speech Recognition on One CPU Core**

> **Bottom Line / Key Takeaways**
> - **What Isosyll does:** Turns live speech into text on **3.1% of one ordinary CPU core** per stream, with no cloud round trip, no GPU and no Python stack. One signed ~1.6 MB program for Linux (x86-64, arm64) and Apple-silicon Macs.
> - **Measured advantage:** Against Zipformer INT8, the strongest open streaming recognizer we measured, Isosyll delivers **2.33× the throughput at 57% of the CPU-core energy per second of audio**, timed side by side on the same core. The same fleet can listen to 2.33× as many streams, or the same streams for about half the energy.
> - **Accuracy:** 5.54% of words wrong while streaming live, 5.00% after the end-of-sentence pass — statistically level with Zipformer's 5.25%. On the half of the data not used to pick settings, the live text makes 0.81 points more errors than Zipformer (bounds +0.62 to +1.02).
> - **Honest scope:** Exploratory results on LibriSpeech development data only. The v3.1.0 model inside v4.0.0 missed its pre-registered accuracy gates and was released by owner decision.

## 1. The Business Problem: Always-On Speech Has a Meter That Never Stops

An always-on recognizer — meeting capture, call-centre assist, in-car and wearable assistants, accessibility captions — must process every second of audio, whether anyone is speaking or not. That cost is paid continuously in cloud bills, battery life and heat. Streaming audio to the cloud adds a network round trip, a per-minute bill, and a privacy question every customer and regulator will ask: where does the recording go? Keeping speech on the device answers it, if the device can afford to listen.

## 2. What Isosyll Changes (Plain English)

Isosyll is a compact *streaming* recognizer: it writes words while the person is still talking, then corrects the sentence when they pause.

- **Live:** 32.7× real time (an hour of speech in under two minutes), 3.1% of one CPU core, 109 MB of memory.
- **Final:** the end-of-sentence pass reaches 5.00% word errors at 16.8× real time.
- **Cheaper listening fleets:** 2.33× as many streams per core as the best open streaming rival, or today's load for about half the CPU energy.
- **Private by default:** speech is recognized where it is captured; the recording never has to leave the building.
- **Battery-friendly:** always-on captions and voice commands without claiming the processor or a GPU.

## 3. The Evidence at a Glance

*730-utterance LibriSpeech DEV selection panel (87.7 min), one pinned AMD Zen 2 core, median of three runs, one scorer for every system. Speed above 1× keeps up with live speech; lower word error rate is better.*

| Recognizer (mode) | Speed | Word errors | Energy / audio-s | Memory |
| --- | ---: | ---: | ---: | ---: |
| **Isosyll live** (160 ms, +LM) | **32.7×** | **5.54%** | **0.328 J** | **109 MB** |
| **Isosyll final** (end-of-utterance pass) | **16.8×** | **5.00%** | — | 387 MB |
| Zipformer INT8 (streaming) | 14.0× | 5.25% | 0.573 J | 152 MB |
| Moonshine Tiny (streaming) | 8.2× | 9.90% | — | — |
| Vosk Small (streaming) | 6.9× | 14.90% | — | — |
| Vosk Large§ (streaming) | 4.1× | 8.80% | — | — |
| Nemotron 0.6B, 560 ms§ (streaming) | 2.4× | 3.56% | — | — |
| Parakeet TDT v3 (offline, whole clip) | 6.7× | 3.36% | — | — |
| Whisper.cpp Tiny§ / Base§ | 0.9× / 0.4× | 17.46% / 15.28% | — | — |

*§ Speed timed on a 73-clip subset; word errors on all 730. Energy is idle-subtracted CPU-core joules per second of audio (not wall power), shown only for the pair timed in the same interleaved blocks; other systems ran in separate sessions. Receipts: [`docs/paper/data/`](../paper/data/).*

Isosyll and Zipformer INT8 were timed in the same blocks on the same core; the 2.33× and 57% figures are medians of within-block ratios. Isosyll's final text (5.00%) versus Zipformer (5.25%) has paired bounds that include zero, so the two are statistically level. Larger 0.6-billion-parameter models (Nemotron, Parakeet) make fewer errors at far higher compute.

## 4. What This Is Not — Honest Limits

> - **Exploratory, released by decision:** The v3.1.0 model inside v4.0.0 missed its pre-registered accuracy gates and was released by owner decision; all results are exploratory.
> - **One benchmark, read English:** All accuracy is LibriSpeech development data (audiobooks). No test-set results are claimed; conversational, accented or noisy speech may rank differently.
> - **Live text trails slightly:** On the half of the data not used to pick settings, live Isosyll makes +0.81 points more word errors than Zipformer (bounds +0.62 to +1.02). The final text closes the gap.
> - **Bigger models are more accurate:** 0.6-billion-parameter models make fewer errors, at far higher compute.
> - **Energy scope:** CPU-core energy only, not wall or battery power; no response-time (latency) claim is made.

## 5. How to Pilot & Next Steps

1. **Install and try it today (Day 1):**
   ```sh
   curl -fsSL https://github.com/neverhuman/jsr/releases/latest/download/install.sh | sh
   isosyll transcribe meeting.wav
   ```
2. **Measure on your own audio and hardware (Weeks 1–2):** run it on a sample of your calls, meetings or field recordings, on machines you already own, and compare word errors and CPU load with your current vendor.
3. **Size the business case (Week 3):** convert measured streams per core into fleet size, cloud spend avoided and battery hours.

**Read next:** [CTO Technical Brief](cto-brief.md) · [Paper (PDF)](../paper/paper.pdf) · [Install Guide](../install.md) · [Release Records](../ultra/releases/)
