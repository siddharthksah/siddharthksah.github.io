---
title: "SnapOtter: why I run OCR, transcription, and upscaling with no cloud at all"
date: 2026-08-29
categories:
  - ai-engineering
permalink: /posts/2026/08/snapotter-local-ai-stack/
tags:
  - Self-Hosting
  - Local AI
  - Open Source
---

Most "free" file tools are a privacy tax. Upload a contract to unlock a PDF, a voice memo to get a transcript, family photos to upscale them, and every one of those files lands on someone else's server, feeding someone else's models.

[SnapOtter](https://snapotter.com) is my answer. It is an open source, self-hosted platform with 200+ file tools across five modalities, where the interesting tools are AI-powered and everything runs on your own hardware. OCR, transcription, upscaling, background removal, face restoration, object erase: no API keys, no upload, no telemetry you didn't opt into.

530,000 Docker image pulls later, I feel comfortable saying the itch was widely shared. The [repository](https://github.com/snapotter-hq/SnapOtter) sits at 2.4k stars, the interface speaks 21 languages including right-to-left scripts, and the whole thing is AGPLv3. This post is about the engineering constraint that shaped all of it, and what building local-first AI taught me about where the actual work lives.

## One Docker command had to give you everything

The design rule was a single sentence: `docker run` must produce the entire product. That rule sounds like packaging trivia until AI models enter the picture, at which point it becomes the architecture.

It means the database ships inside the box. Postgres 17 and Redis 8 are embedded in the default image, so the install is genuinely one command, and a three-container Compose stack exists for people who want the pieces separate. It means multi-arch from day one, because half the self-hosting world runs ARM: Raspberry Pis, ARM cloud free tiers, Apple silicon.

And it means every AI feature had to be selected for hardware I don't control. The reference machine in my head was the box in someone's living room that also runs their media library.

## The stack is small models with good manners

Naming the stack matters, because "local AI" often hides behind vagueness. SnapOtter's tools ride on open models that have earned their reputations: [Tesseract](https://github.com/tesseract-ocr/tesseract) for OCR, [faster-whisper](https://github.com/SYSTRAN/faster-whisper) running the small Whisper variant for transcription, [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for upscaling, [rembg](https://github.com/danielgatis/rembg) for background removal, [GFPGAN](https://github.com/TencentARC/GFPGAN) and [CodeFormer](https://github.com/sczhou/CodeFormer) for face restoration, and [LaMa](https://github.com/advimman/lama) for erasing objects from photos, most of it running through [ONNX Runtime](https://onnxruntime.ai/).

Two decisions there carry most of the weight. The first is choosing the small variant on purpose. Whisper's larger models transcribe better on a benchmark, and they also turn a Raspberry Pi into a space heater. The small model finishes a voice memo while you're still naming the file, and for the jobs people bring to a tool like this, that trade wins almost every time.

The second is lazy downloading. Shipping every model in the base image would bloat it past what a home server should tolerate, so models download on first use, with an explicit consent gate in the code before anything fetches. Your disk pays only for the features you touch.

## Most AI needs are small, and that is the point

The unglamorous truth about day-to-day AI is how modest the requests are. Read the text in this scan. Transcribe this meeting. Make this photo bigger without artifacts. Blur the kids' faces before posting. A frontier model is overkill for every job on that list. Decent open models, packaged so someone who has never heard the word "checkpoint" can use them from a browser, cover all of it, with no subscription attached.

That packaging is where the engineering actually lives. Model selection against CPU budgets. Graceful degradation when the hardware is weak. Sane defaults for people who will never open the settings page. The distance between an open-weights release and a tool your parents could use is the last mile, and the last mile is most of the road.

## Distribution decides whether self-hosted software exists

A self-hosted app that is hard to install has a user count of one. SnapOtter is packaged for Unraid, Cloudron, CasaOS, and Umbrel, and listed in the [awesome-selfhosted](https://awesome-selfhosted.net/) directory, because meeting self-hosters inside their own ecosystems beats asking them to visit yours.

Each ecosystem brings its own manifest format, review process, and update cadence. Maintaining four packagings is repetitive, occasionally thankless work, and it is also where a meaningful share of those 530,000 pulls came from. Distribution is a feature. It just never gets a screenshot on the landing page.

## What the pull counter taught me

Stars measure applause; pulls measure deployments. Watching the second number outrun the first by two orders of magnitude rearranged my sense of what open source success looks like. Most people who run SnapOtter will never open an issue, never star the repository, and never tell me it exists on their machine. The software simply works somewhere I will never see, which is the entire promise of local-first tools kept.

If you self-host anything, [try it](https://snapotter.com). It is AGPLv3, the demo costs you nothing but curiosity, and your files stay yours, which was the whole idea.
