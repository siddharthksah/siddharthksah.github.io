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
2. **"What I build"** block — the three flagships with one-line descriptions and links, in this order: Agentic AI for manufacturing (Panasonic) first, then SnapOtter, then DeepSafe. (Owner decision: AI work leads; personal products follow.)
3. Education, condensed to two lines (SUTD MEng AI; BITS Pilani, thesis at Harvard-MIT HST).
4. Recognition, trimmed: WSA Young Innovators Award (Lisbon, presented by Portugal's science minister), Lockheed Martin C-130J RO-RO Challenge ($25,000 winner), Top 30 Under 30 (BITSAA, Hyperloop India), media features.
5. **Selected publications** block — 3–4 hand-picked papers with venue and year, linking to the papers directly. Confirmed-real seed entries: "Freeform liquid 3D printing of soft functional components for soft robotics" (2022) and "A Miniaturized and Portable Optical Tomography Platform for Volumetric Imaging of Engineered Living Systems" (2019). Owner confirms any additions (candidates needing verification: the IPTC 2026 worker-safety AI paper, TENCON 2024 cyber-threat knowledge graph, ThiefGuard 2025).
6. The "Research & Collaborative Network" bullet list becomes one sentence.

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
- **Publications** — external link to Google Scholar, unchanged. (Review found the profile contains auto-added papers by other "Siddharth Kumar"s; owner decided 2026-08-29 to keep the link as is, risk acknowledged. Cleaning the profile remains recommended but blocks nothing.)
- Sidebar keeps GitHub/LinkedIn/Scholar/email icons.

Deletions: all academicpages placeholder files in `_publications/`, `_talks/`, `_teaching/`, `_portfolio/`, and the now-pointless template pages (`_pages/talks.html`, `_pages/teaching.html`, `_pages/portfolio.html`, `_pages/talkmap.html`, `_pages/publications.md`, `_pages/year-archive.html`, plus any other template-demo pages that would render placeholder or empty-collection content). Nothing reachable may show "Paper Title Number 1"-style content or an empty listing.

Posts get two categories: `ai-engineering` (Writing) and `beyond-work` (Beyond Work). The two section pages filter by category.

## Projects page lineup

Based on a full analysis of 114 archived projects (92 in the BITS-era archive on `/Volumes/16TB_Sid`, 22 modern in `~/Desktop/Personal/Projects`).

**Now — Applied AI** (three feature cards, in this order):

1. **Agentic AI for smart factories** (Panasonic) — leads the page (owner decision). Owner briefing 2026-08-29: he builds agentic AI products for manufacturing / smart-factory operations. Because it has no public artifact to link, the card is substantive-but-anonymized (problem shape, scale, stack); owner reviews the final wording for employer safety before publish.
2. **SnapOtter** (snapotter.com) — open-source self-hosted file-processing platform: 200+ tools, local AI (OCR, transcription, upscaling, face blur), REST API, packaged for Unraid/Cloudron/CasaOS.
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

Seed posts (drafted from owner briefings + repo evidence, owner reviews before publish). The owner wants to write about AI primarily, and every existing post is dated 2023 — so **at least one AI-category post must ship with the launch** or the Writing page opens with a three-year-old newest post:

1. AI seed post for Writing (launch blocker) — easiest candidates with artifacts in hand: the SnapOtter local-AI story or Heartcode (LLM tutor bot); an agentic-AI lessons piece if the owner's employer-safe briefing supports it.
2. Homelab tour (Beyond Work) — M2 Max node, 16TB storage, Nginx Proxy Manager, CrowdSec + pf, Pi-hole, Tailscale, Oracle Cloud free-tier ARM offsite failsafe.
3. Repair Kopitiam (Beyond Work) — owner's stories (needs briefing). Context (repairkopitiam.sg): Singapore community repair movement — monthly "repair-together" sessions (last Sunday, 10am–4pm, nine venues) where volunteer repair coaches help people fix electrical, fabric, and mechanical items; tagline "Love your barang? Fix your barang!". The post frames his participation, with specific repair stories from his briefing.

Backlog, AI-first per owner preference: agentic/LLM engineering pieces as material allows · SnapOtter product story · Heartcode · "From bioprinters to AI" career-arc (thesis → BioP) · zero-cost Oracle offsite failsafe · home automation 2016 → homelab 2026 · Vandubbi build story · Smank build story · Lockheed challenge story · Hyperloop India retrospective · "The flying years" (RC-aircraft folders combined).

## Content strategy (from the AI personal-site survey, 2026-08-29)

Patterns adopted from analyzing ~18 leading AI builders' sites (Huyen, Karpathy, Chollet, Howard, Anandkumar, Liu, Husain, Raschka, Willison, Weng, Lambert, Shankar, Swyx, Chintala, Chase, Yan, Albert, Ball):

- **Copy rules** (applied throughout): mission sentence before job title; numbers not adjectives; third parties carry credibility; explicit role labels on projects.
- **Homepage devices**: dated "Recent" news bullets (keeps the site alive between posts — Shankar/Lambert) and a "Note for AI agents" block (Shankar; on-brand for an agentic-AI engineer).
- **Writing model**: Lilian Weng's — rare but definitive posts on the smart-factory-agents niche, revised over time with dated "Updated:" notes, rather than cadence pressure.
- **Employer-safe writing** (the Panasonic rule): abstraction, not disclaimers — publish industry-level "ghost knowledge" ("what deploying agents on factory floors taught me about evals"), never internals; converting sanitized internal notes into public essays with provenance stated is fine (Yan/Ball). No "views are my own" boilerplate.
- **Long game**: coin and consistently reuse one term for his niche (Chase's "context engineering" move) — candidate to be developed in his first agentic-AI essay.
- **Deferred until the corpus grows**: a "Start Here" page mapping posts into named reading paths (Yan/Raschka); a cheap "notes" second post type for cadence (Willison/Raschka/Ball); ★ favorite flags in post listings (Ball).

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
- Check: all nav links resolve; old post URLs unchanged; no placeholder content reachable (grep built `_site/` for template strings like "Paper Title Number"); no empty-collection or ghost archive pages reachable; avatar renders; email correct everywhere.
- Deploy: push to `master` as `siddharthksah` only after owner sign-off on content.

## Owner prerequisites (block implementation steps that depend on them)

- **Confirm publication list**: which Scholar entries are actually his, beyond the two confirmed papers. Only blocks adding extra entries to the Selected Publications block — the block ships with the two confirmed papers otherwise.
- **Repair Kopitiam briefing**: personal repair stories. Blocks that post only.
- Resolved 2026-08-29: Panasonic briefing received (agentic AI products for smart factories); Scholar link kept as is by owner decision.

## Implementation phases

1. **Structure**: deletions, nav, category pages, config/sidebar changes, photo, SEO. Also update CLAUDE.md's content-model section to match the new structure.
2. **Projects page**: flagship cards + earlier-builds timeline.
3. **Homepage rewrite** (incl. Selected Publications block).
4. **Seed content**: one AI-category Writing post (launch blocker), homelab post (from repo, scrubbed), Repair Kopitiam post.
5. **Ongoing**: article backlog, one at a time, owner-approved.

**Deploy gate:** push to `master` deploys instantly, so phases 1–3 plus the AI seed post go live in a single push after owner sign-off. Deploying phase 1 alone would put nav links to nonexistent pages on the live site. The remaining seed posts may trail.
