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

New nav (`_data/navigation.yml`): **Projects · Writing · Beyond Work**

- **Projects** (`/projects/`, new internal page) — see lineup below.
- **Writing** (`/writing/`, internal page) — lists the AI/engineering posts. The eight existing 2023 posts get a category and surface here; their URLs must not change.
- **Beyond Work** (`/beyond-work/`, new internal page) — homelab, Repair Kopitiam, self-hosting content, kept separate from the professional content.
- **No Publications nav item at launch.** The owner's Google Scholar profile is polluted with auto-added papers by other "Siddharth Kumar"s (verified 2026-08-29: includes a 1989 paper and unrelated fields; the 502-citation count is mostly not his). Linking it would undermine credibility. The homepage Selected Publications block covers publications instead. A Scholar nav link may return only after the owner cleans the profile (delete others' papers, disable automatic article additions).
- Sidebar keeps GitHub/LinkedIn/email icons; the Scholar sidebar icon is likewise gated on profile cleanup.

Deletions: all academicpages placeholder files in `_publications/`, `_talks/`, `_teaching/`, `_portfolio/`, and the now-pointless template pages (`_pages/talks.html`, `_pages/teaching.html`, `_pages/portfolio.html`, `_pages/talkmap.html`, `_pages/publications.md`, `_pages/year-archive.html`, plus any other template-demo pages that would render placeholder or empty-collection content). Nothing reachable may show "Paper Title Number 1"-style content or an empty listing.

Posts get two categories: `ai-engineering` (Writing) and `beyond-work` (Beyond Work). The two section pages filter by category.

## Projects page lineup

Based on a full analysis of 114 archived projects (92 in the BITS-era archive on `/Volumes/16TB_Sid`, 22 modern in `~/Desktop/Personal/Projects`).

**Now — Applied AI** (three feature cards, in this order):

1. **Agentic AI for manufacturing** (Panasonic) — leads the page (owner decision). Because it has no public artifact to link, the card must be substantive-but-anonymized: problem shape, scale, and stack, at a level of detail the owner confirms is employer-safe. Requires an owner briefing before it can be written.
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
3. Repair Kopitiam (Beyond Work) — owner's stories (needs briefing).

Backlog, AI-first per owner preference: agentic/LLM engineering pieces as material allows · SnapOtter product story · Heartcode · "From bioprinters to AI" career-arc (thesis → BioP) · zero-cost Oracle offsite failsafe · home automation 2016 → homelab 2026 · Vandubbi build story · Smank build story · Lockheed challenge story · Hyperloop India retrospective · "The flying years" (RC-aircraft folders combined).

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
- Check: all nav links resolve; old post URLs unchanged; no placeholder content reachable (grep built `_site/` for template strings like "Paper Title Number"); no empty-collection or ghost archive pages reachable; avatar renders; email correct everywhere; no Scholar links anywhere until the profile is cleaned.
- Deploy: push to `master` as `siddharthksah` only after owner sign-off on content.

## Owner prerequisites (block implementation steps that depend on them)

- **Clean the Google Scholar profile**: delete papers that aren't his, disable automatic article additions. Blocks any Scholar link returning to nav/sidebar. (Site can launch without it.)
- **Panasonic briefing**: what the agentic-AI work is and what's employer-safe to say. Blocks the lead Projects card and homepage flagship line.
- **Confirm publication list**: which Scholar entries are actually his, beyond the two confirmed papers. Blocks the Selected Publications block's final content.
- **Repair Kopitiam briefing**: stories for the post. Blocks that post only.

## Implementation phases

1. **Structure**: deletions, nav, category pages, config/sidebar changes, photo, SEO. Also update CLAUDE.md's content-model section to match the new structure.
2. **Projects page**: flagship cards + earlier-builds timeline.
3. **Homepage rewrite** (incl. Selected Publications block).
4. **Seed content**: one AI-category Writing post (launch blocker), homelab post (from repo, scrubbed), Repair Kopitiam post.
5. **Ongoing**: article backlog, one at a time, owner-approved.

**Deploy gate:** push to `master` deploys instantly, so phases 1–3 plus the AI seed post go live in a single push after owner sign-off. Deploying phase 1 alone would put nav links to nonexistent pages on the live site. The remaining seed posts may trail.
