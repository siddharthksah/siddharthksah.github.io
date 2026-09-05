---
title: "Shipping AI models to self-hosted machines"
date: 2026-07-26
categories:
  - ai-engineering
permalink: /posts/2026/07/shipping-ai-models/
tags:
  - Local AI
  - Self-Hosting
  - Open Source
  - MLOps
---

Most AI ships behind an API. The model lives on hardware the operator controls, upgrades happen in one place, and when something breaks, an engineer with dashboards is already looking at it.

[SnapOtter](https://github.com/snapotter-hq/SnapOtter) ships the other way. The models travel to the user: onto a Raspberry Pi behind a TV, a decade-old office tower in a garage, a NAS mounted over a network share that drops packets when the microwave runs. More than half a million Docker pulls means the code now runs on a fleet I will never see, profile, or SSH into.

This post is about what that distribution model did to the engineering. The model itself turned out to be the smallest part. The hard parts are the four stages around it (download, install, load, run), and each one produced a bug report I still think about.

## Downloading models

The first policy decision sounds paranoid and turned out to be right. The app never downloads a model without explicit permission. `SNAPOTTER_ALLOW_MODEL_DOWNLOAD` defaults to off, `HF_HUB_OFFLINE` is forced at startup, and a tool whose model is missing raises an actionable error instead of quietly fetching half a gigabyte. Self-hosters run these boxes on metered connections and [air-gapped](https://en.wikipedia.org/wiki/Air_gap_(networking)) LANs, and a surprise download breaks the local-first promise. Consent gets lifted only inside the [bundle installer](https://github.com/snapotter-hq/SnapOtter/blob/main/packages/ai/python/install_feature.py), only while it runs.

Downloads themselves assume failure. Multi-gigabyte bundles resume across crashes via sidecar metadata files, and every fetch ends in a [SHA256](https://en.wikipedia.org/wiki/SHA-2) check. That's how I found [issue #714](https://github.com/snapotter-hq/SnapOtter/issues/714), my favorite bug of the project. Users on network-mounted storage reported corrupted bundles, and the cause was the accelerated download client. Its parallel writes interleave badly on some NFS and SMB mounts, so the file ends up the right size with the wrong contents. The fix: on checksum mismatch, throw the fast client away and re-download sequentially. When you don't know the filesystem, use the simple path.

## Installing bundles

A feature bundle is a tarball of model weights plus Python packages that has to merge into a live virtual environment, on any filesystem, and survive a power cut at any byte. That sentence took months to make true.

Extraction refuses symlinks, hardlinks, device nodes, and [path traversal](https://en.wikipedia.org/wiki/Directory_traversal_attack) outright. The merge quarantines superseded package versions, so a failed verification can roll back to a known-good state.

A breadcrumb marker sits on disk during the destructive window. If the box dies mid-merge, the next boot sees the marker and rebuilds the environment from scratch. And because `/data` might be a different filesystem from the staging directory, the installer detects cross-filesystem moves and switches to copy-then-rename, since atomic rename only works within one filesystem.

The bug I bring up most is [issue #490](https://github.com/snapotter-hq/SnapOtter/issues/490). The CPU and GPU builds of ONNX Runtime unpack into the same package directory, and a routine bundle update could merge the CPU flavor over the GPU one. Nothing crashed. Inference just got quietly slower for GPU users, which is the worst kind of failure because nobody sees it except as slowness. The rule that fixed it is called "GPU wins" in the code, and the installer enforces it whenever the two flavors collide.

## Loading: a segfault from the shared venv

The AI scripts run in a long-lived [Python dispatcher](https://github.com/snapotter-hq/SnapOtter/blob/main/packages/ai/python/dispatcher.py) that keeps heavy imports warm, gates scripts through a hardcoded allowlist, and cleans up GPU memory after every job. All those scripts share one virtual environment, and that sharing hides a trap.

A running job has native libraries from that venv mapped into its address space via [`dlopen`](https://man7.org/linux/man-pages/man3/dlopen.3.html). A bundle install rewrites those same `.so` files on disk. Let both happen at once and the process can read code that changed under its feet, which ends in a segfault with a stack trace pointing nowhere useful. The fix is a [reader-writer lock](https://github.com/snapotter-hq/SnapOtter/blob/main/packages/ai/src/venv-lock.ts) spanning both runtimes. AI jobs acquire read, installs acquire exclusive write, and the writer is preferred so a queue of jobs can never starve an upgrade. Databases [solved this problem](https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock) decades ago. It just never occurred to me that a folder of Python packages would need the same treatment as a table. The venv is shared mutable state and has to be treated that way.

## Running on unknown hardware

On hardware you can't see, capability detection has to be pessimistic and layered. [The GPU probe](https://github.com/snapotter-hq/SnapOtter/blob/main/packages/ai/python/gpu.py) checks torch, then ONNX Runtime providers, then `nvidia-smi`, and each backend gets its own probe because a GPU that torch can use is no guarantee that [CTranslate2](https://github.com/OpenNMT/CTranslate2) can. When CUDA initialization fails at session time anyway, the code logs it and retries on CPU, because a slow answer beats a dead worker on somebody's NAS.

The fallbacks are explicit down to the numerics: float16 on CUDA, [int8 quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) on CPU, and the [faster-whisper](https://github.com/SYSTRAN/faster-whisper) small model chosen over its benchmark-winning siblings. Animated GIFs taught the warm-up lesson. Instantiating a fresh background-removal session per frame thrashed memory on small boxes, so one [ONNX Runtime](https://onnxruntime.ai/) session now serves every frame of the animation. And when a job dies with exit code 137 regardless, the system classifies the [out-of-memory](https://en.wikipedia.org/wiki/Out_of_memory) kill as an operational condition, which changes the retry decision entirely.

Then ARM64 doubled every one of these problems. Half the self-hosting world runs on it, and the wheels ecosystem still treats it as second class. [mediapipe](https://github.com/google-ai-edge/mediapipe) face detection broke against [protobuf](https://protobuf.dev/) 5 on ARM and stays pinned below it, and one upscaler is skipped in the ARM smoke tests entirely because its stack refuses to cooperate there. Supporting a second architecture multiplies the test matrix at exactly its weakest points.

## What I learned

Four things. Fail closed on anything that touches the network, because someone runs your code with no network at all. Assume the disk lies, checksum everything, and keep a simple sequential path behind every clever fast one. Treat a shared virtual environment like the concurrent mutable state it is, locks included. And pick the small model, because the benchmark chart wasn't run on a Raspberry Pi.

I've made a version of this argument [about factories](https://siddharthksah.github.io/posts/2026/05/factory-grade-agents/) too, and it holds here for a gentler reason. On other people's hardware, the model is the interchangeable part and the machinery around it is the product. Most of the code that made those half a million pulls possible never shows up in a demo.
