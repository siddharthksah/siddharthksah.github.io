---
title: "My homelab: a MacBook, 16 terabytes, and a free ARM server in the cloud"
date: 2026-08-29
categories:
  - beyond-work
permalink: /posts/2026/08/homelab-tour/
tags:
  - Homelab
  - Self-Hosting
  - Networking
---

My homelab is deliberately boring hardware running deliberately paranoid software.

**The node:** an M2 Max MacBook Pro (64 GB RAM) with a 16 TB external drive, wired into the router. Laptops make underrated home servers: built-in UPS, silent, absurd performance per watt, and this one was already paid for.

**The stack:** Docker for almost everything, fronted by Nginx Proxy Manager for routing and certificates. DNS goes through Pi-hole, remote access through Tailscale — nothing is port-forwarded to the open internet that doesn't absolutely need to be.

**The part I'm proudest of:** intrusion handling. Docker Desktop on macOS NATs away real client IPs, which makes fail2ban-style tooling useless inside containers. So a native nginx sits *outside* Docker as the first hop, where real client IPs are visible; CrowdSec reads its logs and bans offenders at the kernel firewall (pf) level. Layered like an onion, and every layer has actually caught things.

**The failsafe:** an always-free ARM VM at a cloud provider — 4 cores, 24 GB RAM, $0/month — running as an off-site node: backups, an external monitor that tells me when home is down (a dead man's switch, since a dead server can't email you), and a Tailscale exit node when I travel. A budget alert guards the "always free" assumption.

**What it actually runs** ranges from media and photo services to [SnapOtter](https://snapotter.com), which I both develop and dogfood here. The lab is also the test bed for the packaging work I do for self-hosting ecosystems — if an update breaks, it breaks in my living room first.

The homelab habit started in 2016 with an Arduino, an RF remote, and a motorized door lock in my university dorm. The hardware got better; the itch is the same.
