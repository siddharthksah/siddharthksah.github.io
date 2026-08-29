# Personal Site Repositioning — Design Spec

**Date:** 2026-08-29
**Goal:** Reposition siddharthksah.github.io so that anyone who Googles Siddharth Kumar Sah comes away thinking "serious AI engineer in Singapore." Audience is general (recruiters, clients/collaborators, and the AI community, roughly equally).

## Scope and non-goals

- **Keep the current theme unchanged**: minimal-mistakes/academicpages look, layout, and the typed.js animated greeting stay as they are. No visual redesign.
- All changes are content, structure, and small trust/SEO details.
- The site stays on Jekyll + GitHub Pages native build; push to `master` deploys. Push using the `siddharthksah` GitHub account (the `snapotter-hq` account is denied on this repo).
- Blog cadence is occasional: seed an initial batch of posts, then publish only when the owner has something to share. The site must not look stale between posts, so evergreen pages (home, Projects) lead.

## Homepage (`_pages/about.md`)

Rewrite to lead with what he builds, not where he studied. Order:

1. Positioning paragraph — Senior AI Engineer II at Panasonic Singapore, bridging applied AI research and production (tightened version of current copy).
2. **"What I build"** block — the three flagships with one-line descriptions and links: SnapOtter, Agentic AI for manufacturing (Panasonic), DeepSafe.
3. Education, condensed to two lines (SUTD MEng AI; BITS Pilani, thesis at Harvard-MIT HST).
4. Recognition, trimmed: WSA Young Innovators Award (Lisbon, presented by Portugal's science minister), Lockheed Martin C-130J RO-RO Challenge ($25,000 winner), Top 30 Under 30 (BITSAA, Hyperloop India), media features.
5. The "Research & Collaborative Network" bullet list becomes one sentence.

Sidebar (author profile in `_config.yml`):

- Avatar restored, using the photo at `/Users/sidd/Documents/profile_photo.jpg`, copied into `images/`.
- Bio line: "Senior AI Engineer · Applied AI research → production" (2–3 variants offered at implementation; owner picks).
- Employer: Panasonic R&D Center Singapore. Location already reads Singapore.
- Email: `siddharthksah@gmail.com` (replaces the obfuscated `firstname+123sk@gmail.com` everywhere it appears).

## Navigation and structure

New nav (`_data/navigation.yml`): **Projects · Writing · Beyond Work · Publications**

- **Projects** (`/projects/`, new internal page) — see lineup below.
- **Writing** (`/writing/`, internal page) — lists the AI/engineering posts. The eight existing 2023 posts get a category and surface here; their URLs must not change.
- **Beyond Work** (`/beyond-work/`, new internal page) — homelab, Repair Kopitiam, self-hosting content, kept separate from the professional content.
- **Publications** — external link to Google Scholar (unchanged behavior, kept deliberately).
- Sidebar keeps GitHub/LinkedIn/Scholar/email icons.

Deletions: all academicpages placeholder files in `_publications/`, `_talks/`, `_teaching/`, `_portfolio/`, and the now-pointless archive pages (`_pages/talks.html`, `_pages/teaching.html`, `_pages/portfolio.html`, `_pages/talkmap.html`, plus any other template-demo pages that would render placeholder content). Nothing reachable may show "Paper Title Number 1"-style content.

Posts get two categories: `ai-engineering` (Writing) and `beyond-work` (Beyond Work). The two section pages filter by category.

## Projects page lineup

Based on a full analysis of 114 archived projects (92 in the BITS-era archive on `/Volumes/16TB_Sid`, 22 modern in `~/Desktop/Personal/Projects`).

**Now — Applied AI** (three feature cards):

1. **SnapOtter** (snapotter.com) — open-source self-hosted file-processing platform: 200+ tools, local AI (OCR, transcription, upscaling, face blur), REST API, packaged for Unraid/Cloudron/CasaOS.
2. **Agentic AI for manufacturing** — Panasonic work, described at whatever level of detail the employer allows.
3. **DeepSafe** — open-source deepfake detection platform (his own project; the `deepsafehq` org is his).

**Earlier builds** (compact timeline, one to two lines each):

| Entry | Note |
|---|---|
| Harvard-MIT thesis | Bioprinting research, Khademhosseini Lab; thesis PDF survives |
| BioP | 3D bioprinter: custom bio-inks, firmware, CAD, research paper; carries the WSA award story. Named "BioP" (not "BioP India"); the catheter-printer work is part of BioP, not a separate entry |
| Lockheed Martin C-130J RO-RO Challenge | $25,000 winner; disaster-relief logistics, water filtration, drone command-and-control |
| Hyperloop India | Proposals and final design report with CFD work |
| Vandubbi | Underwater ROV: FPV, 4 kg payload, 300 m range, auto-surface failsafe; Quark 2017 |
| Smank | InMoov-based humanoid robot, 100+ printed parts |
| Ciclop 3D scanner | 2016 laser scanner build |
| PROJECT HOMON | 2016 Arduino home automation, framed as the homelab origin |

Explicitly removed: Biàn (a toilet-design project, misidentified by analysis as a robotic arm — owner correction). Honourable mentions (article-only, not on the page): Chem-E-Car, IAFSM gripper, Women Safety wearable, bioimaging scanner, XY plotter, Bitsian Browser, HIRA voice assistant.

## Beyond Work content plan

Seed posts (drafted from owner briefings + repo evidence, owner reviews before publish):

1. Homelab tour — M2 Max node, 16TB storage, Nginx Proxy Manager, CrowdSec + pf, Pi-hole, Tailscale, Oracle Cloud free-tier ARM offsite failsafe.
2. Repair Kopitiam — owner's stories (needs briefing).

Backlog, in publish order: SnapOtter product story · "From bioprinters to AI" career-arc (thesis → BioP) · Vandubbi build story · Smank build story · Lockheed challenge story · Hyperloop India retrospective · "The flying years" (RC-aircraft folders combined) · Heartcode (Telegram LeetCode tutor bot) · zero-cost Oracle offsite failsafe · home automation 2016 → homelab 2026.

## Safety and privacy constraints

- **autogit never appears on the site** (a commit-activity bot would undercut credibility).
- Homelab/networking articles must scrub: static IP, DuckDNS domain, internal IPs/topology, and any hostnames from the repos.
- Project Holy is under NDA — never referenced. Personal folders (Pics, dj, Ayushi, Gift, applications, Majulah Dealz, sell-laptops) stay untouched and unmentioned.
- Panasonic work described only at a publicly shareable level; owner confirms wording.

## SEO and discoverability

- `_config.yml`: site title "Siddharth Kumar Sah — Senior AI Engineer in Singapore", proper description, correct `url`.
- JSON-LD Person markup (name, jobTitle, worksFor, sameAs: GitHub/LinkedIn/Scholar) in `_includes/head/custom.html`.
- `docs/` added to Jekyll `exclude:` so spec/plan documents never publish to the live site.

## Verification

- Local: `bundle exec jekyll serve --config _config.yml,_config.dev.yml`; restart after `_config.yml` edits.
- Check: all four nav links resolve; old post URLs unchanged; no placeholder content reachable (grep built `_site/` for template strings like "Paper Title Number"); avatar renders; email correct everywhere.
- Deploy: push to `master` as `siddharthksah` only after owner sign-off on content.

## Implementation phases

1. **Structure**: deletions, nav, category pages, config/sidebar changes, photo, SEO. Ships first.
2. **Projects page**: flagship cards + earlier-builds timeline.
3. **Homepage rewrite**.
4. **Seed content**: homelab post (from repo, scrubbed), Repair Kopitiam post (needs owner briefing).
5. **Ongoing**: article backlog, one at a time, owner-approved.
