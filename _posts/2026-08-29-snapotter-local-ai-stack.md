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

Most "free" file tools are a privacy tax. Upload a contract to unlock a PDF, a voice memo to get a transcript, family photos to upscale them — and every one of those files lands on someone else's server, feeding someone else's models.

[SnapOtter](https://snapotter.com) is my answer: an open-source, self-hosted platform with 200+ file tools — images, video, audio, PDFs — where the interesting ones are AI-powered and **everything runs on your own hardware**. OCR, audio transcription, image upscaling, background removal, face blur, photo restoration: no API keys, no upload, no telemetry you didn't opt into.

## The constraint that shaped everything

The design rule was one line: *a single Docker command must give you the whole thing.*

That rule is harsher than it sounds once AI models enter the picture. It means:

- **Models ship with the container, not behind an API.** Every AI feature had to be selected for CPU-tolerable inference on a home server — the kind of machine that also runs your Jellyfin.
- **One container, batteries included.** Postgres and Redis are embedded in the default image, so `docker run` genuinely is the whole install. A three-container Compose stack exists for people who want to scale the pieces separately.
- **Multi-arch from day one.** Half the self-hosting world runs ARM (Raspberry Pis, ARM cloud free tiers, Apple silicon), so AMD64-only was never an option.

## What "local AI" buys you in practice

The unglamorous truth is that most day-to-day AI needs are small: read the text in this scan, transcribe this meeting, make this photo bigger without artifacts, blur the kids' faces before posting. None of that needs a frontier model or a GPU cluster — it needs decent open models, packaged so a non-ML person can use them from a browser.

That packaging is the actual engineering. Model selection against CPU budgets, graceful degradation when hardware is weak, sane defaults for people who will never read the settings page — the last mile between an open-weights checkpoint and a tool your parents could use.

## Distribution is a feature

A self-hosted app that is hard to install does not exist. SnapOtter is packaged for Unraid, Cloudron, CasaOS, and Umbrel — going to where self-hosters already are instead of asking them to come to you. Each ecosystem has its own manifest format, review process, and update cadence; maintaining them is boring and it is also why people actually run the thing.

If you self-host anything, [try it](https://snapotter.com) — it is AGPLv3, and the demo instance costs you nothing but curiosity.
