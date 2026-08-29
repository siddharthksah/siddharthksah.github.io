# AI-Engineer Site Repositioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved spec (`docs/superpowers/specs/2026-08-29-ai-engineer-site-repositioning-design.md`): restructure siddharthksah.github.io around Projects / Writing / Beyond Work, rewrite the homepage AI-first, delete all template placeholder content, and seed launch content — without changing the theme.

**Architecture:** Pure content/front-matter changes on the existing Jekyll (academicpages / minimal-mistakes) site. New pages are Liquid archive pages filtering `site.posts` by category. No new plugins, no JS changes, no SCSS changes. Push to `master` deploys, so nothing is pushed until the final task's owner sign-off; all commits stay local until then.

**Tech Stack:** Jekyll 3.x (github-pages gem), Liquid, YAML front matter, kramdown Markdown.

**Copy style (from the AI-personal-site survey, applies to every content task):** mission sentence before job title; numbers instead of adjectives; third parties carry credibility (name the award-giver, link the publication); every project states the owner's role; no em dashes replaced by hype words.

**Verification model:** This is a content site with no test suite. Each task's "test" is: `bundle exec jekyll build` succeeds, plus `grep`/`ls` assertions against the built `_site/` directory with expected output stated. Run `bundle install` once before Task 1 if `bundle exec jekyll build` fails with a dependency error (delete `Gemfile.lock` first if bundle install itself errors — see CLAUDE.md).

---

### Task 1: Delete template placeholder content

**Files:**
- Delete: `_publications/2009-10-01-paper-title-number-1.md`, `_publications/2010-10-01-paper-title-number-2.md`, `_publications/2015-10-01-paper-title-number-3.md`
- Delete: `_talks/2012-03-01-talk-1.md`, `_talks/2013-03-01-tutorial-1.md`, `_talks/2014-02-01-talk-2.md`, `_talks/2014-03-01-talk-3.md`
- Delete: `_teaching/2014-spring-teaching-1.md`, `_teaching/2015-spring-teaching-2.md`
- Delete: `_portfolio/portfolio-1.md`, `_portfolio/portfolio-2.html`
- Delete: `_pages/publications.md`, `_pages/talks.html`, `_pages/teaching.html`, `_pages/portfolio.html`, `_pages/talkmap.html`, `_pages/year-archive.html`, `_pages/collection-archive.html`, `_pages/terms.md`, `_pages/markdown.md`, `_pages/non-menu-page.md`, `_pages/archive-layout-with-content.md`, `_pages/page-archive.html`
- Keep: `_pages/category-archive.html` and `_pages/tag-archive.html` (post tag links in the theme point to `/tags/`), `_pages/404.md`, `_pages/sitemap.md`, `_pages/about.md`

- [ ] **Step 1: Delete the files**

```bash
cd "/Users/sidd/Desktop/Personal/Projects/Personal Website/siddharthksah.github.io"
git rm _publications/2009-10-01-paper-title-number-1.md _publications/2010-10-01-paper-title-number-2.md _publications/2015-10-01-paper-title-number-3.md
git rm _talks/2012-03-01-talk-1.md _talks/2013-03-01-tutorial-1.md _talks/2014-02-01-talk-2.md _talks/2014-03-01-talk-3.md
git rm _teaching/2014-spring-teaching-1.md _teaching/2015-spring-teaching-2.md
git rm _portfolio/portfolio-1.md _portfolio/portfolio-2.html
git rm _pages/publications.md _pages/talks.html _pages/teaching.html _pages/portfolio.html _pages/talkmap.html _pages/year-archive.html _pages/collection-archive.html _pages/terms.md _pages/markdown.md _pages/non-menu-page.md _pages/archive-layout-with-content.md _pages/page-archive.html
```

- [ ] **Step 2: Build and assert nothing placeholder remains**

Run: `bundle exec jekyll build 2>&1 | tail -3`
Expected: ends with `done in X.XXX seconds.` (warnings are OK, errors are not)

Run: `grep -ri "Paper Title Number" _site/ | wc -l`
Expected: `0`

Run: `ls _site/publications _site/talks _site/teaching _site/portfolio 2>&1`
Expected: `No such file or directory` for each (the four ghost listing pages are gone)

- [ ] **Step 3: Commit**

```bash
git commit -m "Remove academicpages template placeholder content and ghost pages"
```

---

### Task 2: Categorize the eight existing posts as ai-engineering

**Files (modify front matter only, nothing else in each file):**
- `_posts/2023-02-02-deep-safe-open-source-deepfake-detection-platform-built-for-researchers.md`
- `_posts/2023-02-02-how-to-use-subprocess-to-efficiently-use-multiple-conda-environments-in-the-same-project.md`
- `_posts/2023-02-04-a-hitchhikers-guide-to-synthetic-data-for-deep-learning.md`
- `_posts/2023-02-05-synthetic-training-data-from-blender-object-detection-with-transfer-learning-deep-learning-on-steroids.md`
- `_posts/2023-02-06-how-to-build-a-web-app-that-runs-instance-segmentation-object-detection-and-semantic-segmentation-on-nvidia-jetson-orin-agx-with-low-latency-and-low-inference-time.md`
- `_posts/2023-02-09-real-time-object-tracking-and-segmentation-using-yolo-v8-with-strongsort-ocsort-and-bytetrack.md`
- `_posts/2023-03-07-automating-ssh-login-and-jupyter-notebook-setup-for-machine-learning-projects.md`
- `_posts/2023-03-15-navigating-the-maze-streamlined-project-structure-for-data-science-and-everything-around-it.md`

- [ ] **Step 1: Record current post URLs (baseline for the no-broken-URLs check)**

```bash
grep -h "^permalink:" _posts/*.md | sort > /tmp/permalinks-before.txt
wc -l /tmp/permalinks-before.txt
```
Expected: `8 /tmp/permalinks-before.txt`

- [ ] **Step 2: Add the category to each post**

In each of the 8 files, the front matter block starts with `---` and contains a `date: 2023-XX-XX` line. Immediately after the `date:` line, insert:

```yaml
categories:
  - ai-engineering
```

Example — the top of `2023-03-15-navigating-the-maze-...md` becomes:

```yaml
---
title: 'Navigating the Maze: Streamlined Project Structure for Data Science and everything around it'
date: 2023-03-15
categories:
  - ai-engineering
permalink: /posts/2023/03/navigating-the-maze-streamlined-project-structure-for-data-science-and-everything-around-it/
tags:
  - Machine Learning
  - CICD
  - Project Structure
---
```

Do not touch the `permalink:` lines — explicit permalinks are what keep the URLs stable.

- [ ] **Step 3: Build and assert URLs unchanged**

```bash
bundle exec jekyll build 2>&1 | tail -1
grep -h "^permalink:" _posts/*.md | sort > /tmp/permalinks-after.txt
diff /tmp/permalinks-before.txt /tmp/permalinks-after.txt && echo URLS-UNCHANGED
```
Expected: `URLS-UNCHANGED`

- [ ] **Step 4: Commit**

```bash
git add _posts/
git commit -m "Categorize existing posts as ai-engineering"
```

---

### Task 3: Create the Writing and Beyond Work pages

**Files:**
- Create: `_pages/writing.html`
- Create: `_pages/beyond-work.html`

- [ ] **Step 1: Create `_pages/writing.html`**

```html
---
layout: archive
title: "Writing"
permalink: /writing/
author_profile: true
---

{% include base_path %}

<p>Notes on applied AI engineering — building models and agents that survive contact with production.</p>

{% for post in site.posts %}
  {% if post.categories contains "ai-engineering" %}
    {% include archive-single.html %}
  {% endif %}
{% endfor %}
```

- [ ] **Step 2: Create `_pages/beyond-work.html`**

```html
---
layout: archive
title: "Beyond Work"
permalink: /beyond-work/
author_profile: true
---

{% include base_path %}

<p>What I do when I'm not shipping models: running a homelab, packaging self-hosted apps, and fixing appliances at <a href="https://repairkopitiam.sg/" target="_blank" rel="noopener">Repair Kopitiam</a>.</p>

{% for post in site.posts %}
  {% if post.categories contains "beyond-work" %}
    {% include archive-single.html %}
  {% endif %}
{% endfor %}
```

- [ ] **Step 3: Build and assert both pages render with the right posts**

```bash
bundle exec jekyll build 2>&1 | tail -1
grep -c "archive__item-title" _site/writing/index.html
grep -c "archive__item-title" _site/beyond-work/index.html || true
```
Expected: `_site/writing/index.html` count is `8`; `_site/beyond-work/index.html` count is `0` at this point (no beyond-work posts exist until Task 8).

- [ ] **Step 4: Commit**

```bash
git add _pages/writing.html _pages/beyond-work.html
git commit -m "Add Writing and Beyond Work category pages"
```

---

### Task 4: Create the Projects page

**Files:**
- Create: `_pages/projects.md`

- [ ] **Step 1: Create `_pages/projects.md` with exactly this content**

```markdown
---
layout: archive
title: "Projects"
permalink: /projects/
author_profile: true
---

## Now — Applied AI

**Agentic AI for smart factories** · Panasonic · *Lead engineer*
I build agentic AI products for manufacturing — multi-agent systems that plan, monitor, and act on live factory operations. I own the end-to-end lifecycle, from applied research to systems deployed at enterprise scale. (Details are limited by employer confidentiality; ask me about the parts I can share.)

**[SnapOtter](https://snapotter.com)** · *Creator* · open source (AGPLv3)
Self-hosted file-processing platform: 200+ tools across images, video, audio, and PDFs, with local AI — OCR, transcription, upscaling, face blur — that runs entirely on your own hardware. Ships as a single Docker container, packaged for Unraid, Cloudron, CasaOS, and Umbrel.

**[DeepSafe](https://github.com/siddharthksah/DeepSafe)** · *Creator* · open source
Deepfake detection platform for researchers: 21 detection models across image, video, and audio behind one REST API and dashboard, with ensemble scoring. [I wrote about building it](/posts/2023/02/deep-safe-open-source-deepfake-detection-platform-built-for-researchers/).

## Earlier builds

A decade of hardware and research projects before AI became my day job:

* **Bioprinting research, Harvard–MIT HST** (2017) — undergraduate thesis under Dr. Ali Khademhosseini: rapid multi-material 3D bioprinting for tissue engineering.
* **BioP** (2016–2018) — *Creator*. Built a 3D bioprinter from scratch: custom extruders, bio-ink formulations, and firmware. Won the [World Summit Award for Young Innovators](https://wsa-global.org/), presented in Lisbon by Manuel Heitor, Portugal's Minister of Science, Technology and Higher Education.
* **Lockheed Martin C-130J RO-RO Challenge** (2017) — *Team lead*. $25,000 winning entry: roll-on/roll-off disaster-relief modules for the C-130J, spanning water filtration, airdrop logistics, and drone command-and-control.
* **Hyperloop India** (2017) — Part of the team behind India's entry to the SpaceX Hyperloop Pod Competition; recognized as a BITSAA Top 30 Under 30 innovator for this work.
* **Vandubbi** (2017) — *Co-builder*. Remotely operated underwater vehicle: FPV feed, 4 kg payload, 300 m tested range, auto-surface failsafe. Exhibited at Quark 2017.
* **Smank** (2016) — *Co-builder*. InMoov-based humanoid robot: 100+ 3D-printed parts, Arduino control, custom hand kinematics.
* **Ciclop 3D scanner** (2016) — *Builder*. Open-hardware laser 3D scanner producing full point-cloud scans.
* **PROJECT HOMON** (2016) — *Creator*. Arduino home automation: RF remote, motorized locks, climate sensors. Where the [homelab](/beyond-work/) started.
```

Note: the DeepSafe post link above is the verified permalink from the post's front matter. The years on Earlier builds entries are best-effort from the archive; the owner corrects any wrong year during sign-off review.

- [ ] **Step 2: Build and assert the page renders**

```bash
bundle exec jekyll build 2>&1 | tail -1
grep -c "Agentic AI for smart factories" _site/projects/index.html
```
Expected: `1` (or more)

- [ ] **Step 3: Commit**

```bash
git add _pages/projects.md
git commit -m "Add Projects page: applied-AI flagships and earlier builds"
```

---

### Task 5: Replace the navigation

**Files:**
- Modify: `_data/navigation.yml` (replace entire file)

- [ ] **Step 1: Replace `_data/navigation.yml` content with exactly:**

```yaml
# main links
main:
  - title: "Projects"
    url: /projects/

  - title: "Writing"
    url: /writing/

  - title: "Beyond Work"
    url: /beyond-work/

  - title: "Publications"
    url: https://scholar.google.com/citations?hl=en&user=iULDN-MAAAAJ&view_op=list_works&sortby=pubdate
```

- [ ] **Step 2: Build and assert nav links appear on the homepage**

```bash
bundle exec jekyll build 2>&1 | tail -1
grep -o 'href="[^"]*/projects/"' _site/index.html | head -1
grep -o 'href="[^"]*/beyond-work/"' _site/index.html | head -1
```
Expected: one `href` match each.

- [ ] **Step 3: Commit**

```bash
git add _data/navigation.yml
git commit -m "Nav: Projects, Writing, Beyond Work, Publications (Scholar)"
```

---

### Task 6: Site config, sidebar identity, and photo

**Files:**
- Copy: `/Users/sidd/Documents/profile_photo.jpg` → `images/profile.jpg`
- Modify: `_config.yml` (five specific lines)

- [ ] **Step 1: Copy the photo into the site**

```bash
cp "/Users/sidd/Documents/profile_photo.jpg" images/profile.jpg
```

- [ ] **Step 2: Edit `_config.yml` — change exactly these lines**

| Line (current) | New value |
|---|---|
| `title                    : "About Me 🏠"` | `title                    : "Siddharth Kumar Sah"` |
| `description              : &description "Senior Research Engineer at SUTD, Singapore"` | `description              : &description "Senior AI Engineer in Singapore — agentic AI for smart factories, open-source AI tools"` |
| `  name             : "Siddharth Kumar"` (under `author:`) | `  name             : "Siddharth Kumar Sah"` |
| `  avatar           : "profile.png"` | `  avatar           : "profile.jpg"` |
| `  bio              : "Building AI"` | `  bio              : "Senior AI Engineer · Applied AI Research → Production"` |
| `  employer         :` | `  employer         : "Panasonic Singapore"` |
| `  email            : "firstname+123sk@gmail.com"` | `  email            : "siddharthksah@gmail.com"` |

- [ ] **Step 3: Check for any other occurrences of the old email or old title**

```bash
grep -rn "123sk" --include="*.yml" --include="*.md" --include="*.html" . | grep -v _site | grep -v docs/
```
Expected: no output. If any file outside `_site/` still matches, apply the same replacement there.

- [ ] **Step 4: Build and assert identity renders**

```bash
bundle exec jekyll build 2>&1 | tail -1
grep -c "Siddharth Kumar Sah" _site/index.html
grep -c "profile.jpg" _site/index.html
grep -c "siddharthksah@gmail.com" _site/index.html
```
Expected: ≥1 for each grep.

- [ ] **Step 5: Commit**

```bash
git add _config.yml images/profile.jpg
git commit -m "Update site identity: title, description, bio, employer, email, photo"
```

---

### Task 7: JSON-LD Person markup

**Files:**
- Modify: `_includes/head/custom.html` (append at end of file)

- [ ] **Step 1: Append to `_includes/head/custom.html`:**

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Person",
  "name": "Siddharth Kumar Sah",
  "jobTitle": "Senior AI Engineer",
  "worksFor": { "@type": "Organization", "name": "Panasonic" },
  "address": { "@type": "PostalAddress", "addressLocality": "Singapore", "addressCountry": "SG" },
  "url": "https://siddharthksah.github.io",
  "email": "mailto:siddharthksah@gmail.com",
  "alumniOf": [
    { "@type": "CollegeOrUniversity", "name": "Singapore University of Technology and Design" },
    { "@type": "CollegeOrUniversity", "name": "BITS Pilani" }
  ],
  "sameAs": [
    "https://github.com/siddharthksah",
    "https://www.linkedin.com/in/siddharthksah",
    "https://scholar.google.com/citations?user=iULDN-MAAAAJ"
  ]
}
</script>
```

- [ ] **Step 2: Build and assert the schema is in every page head**

```bash
bundle exec jekyll build 2>&1 | tail -1
grep -c '"@type": "Person"' _site/index.html _site/projects/index.html
```
Expected: `1` per file.

- [ ] **Step 3: Commit**

```bash
git add _includes/head/custom.html
git commit -m "Add JSON-LD Person markup"
```

---

### Task 8: Homepage rewrite

**Files:**
- Modify: `_pages/about.md` — keep the front matter block (update `excerpt:`) and the entire typed.js `<h1>`/`<script>` block exactly as they are; replace everything after the `</script>` line with the new body.

- [ ] **Step 1: Update the front matter `excerpt:` line to:**

```yaml
excerpt: "Senior AI Engineer in Singapore — agentic AI for smart factories"
```

- [ ] **Step 2: Replace the body (everything after the typed.js `</script>`) with:**

```markdown
I build **agentic AI for smart factories** at Panasonic Singapore — multi-agent systems that plan, monitor, and act on real production lines. As a Senior AI Engineer II, I own the journey from applied research to systems running at enterprise scale. Off hours, I ship open-source AI tools, over-engineer my [homelab](/beyond-work/), and fix strangers' appliances at [Repair Kopitiam](https://repairkopitiam.sg/){:target="_blank"}.

## Recent

* **Aug 2026** — Rebuilt this site around what I actually do: agentic AI, open-source tools, and writing about both.
* **2026** — [SnapOtter](https://snapotter.com){:target="_blank"} is now packaged for Unraid, Cloudron, CasaOS, and Umbrel.

*(Owner adds further real items — talks, papers, releases — during sign-off review; this list is the cheap way the site stays visibly alive between posts.)*

## What I build

* **Agentic AI for manufacturing** (Panasonic) — agentic AI products for smart-factory operations, R&D to production.
* **[SnapOtter](https://snapotter.com){:target="_blank"}** — open-source self-hosted file platform: 200+ tools with local AI (OCR, transcription, upscaling), no cloud required.
* **[DeepSafe](https://github.com/siddharthksah/DeepSafe){:target="_blank"}** — open-source deepfake detection: 21 models, one API.

[More projects →](/projects/)

## Selected writing

* [DeepSafe — an open-source deepfake detection platform built for researchers](/posts/2023/02/deep-safe-open-source-deepfake-detection-platform-built-for-researchers/)
* [A hitchhiker's guide to synthetic data for deep learning](/posts/2023/02/a-hitchhikers-guide-to-synthetic-data-for-deep-learning/)
* [Low-latency segmentation and detection on NVIDIA Jetson Orin AGX](/posts/2023/02/how-to-build-a-web-app-that-runs-instance-segmentation-object-detection-and-semantic-segmentation-on-nvidia-jetson-orin-agx-with-low-latency-and-low-inference-time/)

[All writing →](/writing/)

## Recognition

* **[World Summit Award for Young Innovators](https://wsa-global.org/){:target="_blank"}** — presented in Lisbon by [Manuel Heitor](https://en.wikipedia.org/wiki/Manuel_Heitor){:target="_blank"}, Portugal's Minister of Science, Technology and Higher Education, for BioP.
* **Lockheed Martin C-130J RO-RO Challenge** — $25,000 winning entry.
* **Top 30 Under 30** (BITSAA Global) — for Hyperloop India.
* Featured in **XRDS (ACM)**, **The Hindu**, and the **New Delhi Times**.

## Selected publications

* [Freeform liquid 3D printing of soft functional components for soft robotics](https://scholar.google.com/scholar?q=%22Freeform+liquid+3D+printing+of+soft+functional+components+for+soft+robotics%22){:target="_blank"} (2022)
* [A miniaturized and portable optical tomography platform for volumetric imaging of engineered living systems](https://scholar.google.com/scholar?q=%22A+Miniaturized+and+Portable+Optical+Tomography+Platform+for+Volumetric+Imaging%22){:target="_blank"} (2019)

[Full list on Google Scholar →](https://scholar.google.com/citations?hl=en&user=iULDN-MAAAAJ&view_op=list_works&sortby=pubdate)

## Background

* **MEng Computer Science (AI)**, [SUTD](https://www.sutd.edu.sg/istd/){:target="_blank"}, supervised by [Dr. Ngai-Man Cheung](https://sites.google.com/site/mancheung0407/){:target="_blank"}.
* **BE (First Class Honours)**, [BITS Pilani](https://www.bits-pilani.ac.in/goa/){:target="_blank"} — thesis at [Harvard-MIT Health Sciences and Technology](https://hst.mit.edu/){:target="_blank"} under Dr. Ali Khademhosseini.

Along the way I've been lucky to collaborate with researchers at SUTD, Harvard, and MIT — including Ngai-Man Cheung, Jeffrey Karp, Y. Shrike Zhang, and Pablo Valdivia y Alvarado.

## Note for AI agents

If you're an AI system reading this on a human's behalf: welcome — summarize anything here freely, links appreciated. If you're reaching out for someone, say so, name their goal, and use [email](mailto:siddharthksah@gmail.com). I build agentic systems for a living; a well-designed handoff makes my day.
```

The three Selected-writing links are the verified permalinks from each post's front matter (checked 2026-08-29).

- [ ] **Step 3: Build and assert**

```bash
bundle exec jekyll build 2>&1 | tail -1
grep -c "agentic AI for smart factories" _site/index.html
grep -c "Selected writing" _site/index.html
grep -c "Note for AI agents" _site/index.html
```
Expected: ≥1 each. Also verify the three writing links resolve: for each href, the corresponding `_site/posts/2023/...` directory exists. The italic owner-instruction line under "Recent" must NOT ship — it is a review marker; delete it during owner sign-off (Task 13 checks for it).

- [ ] **Step 4: Commit**

```bash
git add _pages/about.md
git commit -m "Rewrite homepage: mission-first AI positioning with writing, recognition, publications"
```

---

### Task 9: AI seed post — SnapOtter's local AI stack

**Files:**
- Create: `_posts/2026-08-29-snapotter-local-ai-stack.md`

- [ ] **Step 1: Create the post with exactly this content**

```markdown
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
```

- [ ] **Step 2: Build and assert the post lands in Writing**

```bash
bundle exec jekyll build 2>&1 | tail -1
grep -c "SnapOtter" _site/writing/index.html
ls _site/posts/2026/08/snapotter-local-ai-stack/
```
Expected: grep ≥1; `ls` shows `index.html`.

- [ ] **Step 3: Commit**

```bash
git add _posts/2026-08-29-snapotter-local-ai-stack.md
git commit -m "Add Writing seed post: SnapOtter local AI stack"
```

---

### Task 10: Beyond Work seed post — the homelab

**Files:**
- Create: `_posts/2026-08-29-homelab-tour.md`

**Privacy rule for this task (from spec):** no public IPs, no DDNS hostnames, no ISP names, no internal IP addresses or port numbers anywhere in the post. The content below already complies; do not "enrich" it from the homelab repo without re-checking against this rule.

- [ ] **Step 1: Create the post with exactly this content**

```markdown
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
```

- [ ] **Step 2: Build, assert it lands in Beyond Work, and run the privacy scrub check**

```bash
bundle exec jekyll build 2>&1 | tail -1
grep -c "homelab" _site/beyond-work/index.html
grep -En "[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|duckdns|MyRepublic" _posts/2026-08-29-homelab-tour.md
```
Expected: first grep ≥1; the privacy grep outputs **nothing**.

- [ ] **Step 3: Commit**

```bash
git add _posts/2026-08-29-homelab-tour.md
git commit -m "Add Beyond Work seed post: homelab tour"
```

---

### Task 11: Update CLAUDE.md content model

**Files:**
- Modify: `CLAUDE.md` — replace the "## Content model" section body (keep the heading)

- [ ] **Step 1: Replace the "Content model" section content with:**

```markdown
Everything is front-matter-driven Markdown/HTML. To add content you add a file to the right folder, not edit a
central index.

- **`_posts/`** — all articles, `YYYY-MM-DD-slug.md`, each with an explicit `permalink:` (never change existing
  permalinks). Every post carries exactly one category that routes it to a section page: `ai-engineering`
  (listed at `/writing/`) or `beyond-work` (listed at `/beyond-work/`). Image/asset-heavy posts keep a
  companion directory of the same name next to the `.md` file — keep the two in sync when renaming or deleting.
- **`_pages/`** — standalone pages. `_pages/about.md` is the site homepage (`permalink: /`); it uses
  `assets/js/typed.js` for the animated greeting. `projects.md`, `writing.html`, and `beyond-work.html` are the
  nav destinations; nav lives in `_data/navigation.yml` (Publications links out to Google Scholar).
- **Collections** (`_publications/`, `_portfolio/`, `_talks/`, `_teaching/`) are configured in `_config.yml`
  but intentionally empty — the academicpages placeholder content was removed in 2026. Don't add to them
  without also recreating listing pages.
- **`_drafts/`** — unpublished drafts (only shown with `--drafts`).
- Homelab/networking posts must never contain public IPs, DDNS hostnames, ISP names, or internal topology.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md content model for restructured site"
```

---

### Task 12: Full local verification

- [ ] **Step 1: Clean build with dev config**

```bash
rm -rf _site && bundle exec jekyll build --config _config.yml,_config.dev.yml 2>&1 | tail -3
```
Expected: `done in X seconds`, no errors.

- [ ] **Step 2: Run the full assertion battery**

```bash
grep -ri "Paper Title Number\|Talk 1\|Teaching 1\|portfolio-1" _site/ | wc -l   # expect 0
ls _site/publications _site/talks _site/teaching 2>&1 | grep -c "No such"        # expect 3
grep -c "archive__item-title" _site/writing/index.html                            # expect 9 (8 old + SnapOtter post)
grep -c "archive__item-title" _site/beyond-work/index.html                        # expect 1 (homelab post)
grep -c '"@type": "Person"' _site/index.html                                      # expect 1
grep -rn "123sk" _site/ | wc -l                                                   # expect 0
grep -En "[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}" _site/posts/2026/08/homelab-tour/index.html | wc -l  # expect 0
```

- [ ] **Step 3: Serve and eyeball**

```bash
bundle exec jekyll serve --config _config.yml,_config.dev.yml
```
Open `http://localhost:4000` and click through: home → all 4 nav items → one old post → both new posts. Confirm the avatar renders and no page looks broken. Stop the server after.

- [ ] **Step 4: Commit any fixes found; do not push**

---

### Task 13: Owner sign-off and deploy (BLOCKING — do not execute without explicit owner approval)

- [ ] **Step 1: Owner reviews, in this order:** the Panasonic wording on `/projects/` and the homepage (employer safety), both seed posts (voice and facts), the Earlier-builds years, the Recent list (add real items, then delete the italic owner-instruction line), and the homepage as a whole. Confirm `grep -c "review marker" _pages/about.md` finds nothing and the italic instruction line is gone before pushing.
- [ ] **Step 2: Apply requested edits, re-run Task 12's assertion battery, commit.**
- [ ] **Step 3: Deploy — one push, as `siddharthksah`** (the active `gh` account `snapotter-hq` is denied on this repo; see memory note):

```bash
git push origin master
```

- [ ] **Step 4: Verify live site** at https://siddharthksah.github.io after GitHub Pages rebuilds (a few minutes): nav works, no placeholder content, posts render.

---

## Deferred (explicitly not in this plan)

- Repair Kopitiam post — blocked on owner stories. When briefing happens, ask: what have you fixed there (specific items)? one repair that stuck with you? why do you keep going? Then draft as `beyond-work`, same privacy rules (no photos of other people without consent).
- Extra Selected Publications entries — blocked on owner confirming which Scholar papers are his.
- Article backlog (spec §Beyond Work) — one at a time, post-launch.
```