# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Personal academic website (`siddharthksah.github.io`) — a Jekyll site forked (and detached) from
[academicpages](https://github.com/academicpages/academicpages.github.io), which is itself a fork of the
[Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) theme. It is served by GitHub Pages'
**native build** (the `github-pages` gem), so there is no test suite, no CI workflow, and no build artifact to
commit — pushing to `master` is the deploy.

## Commands

```bash
bundle install                    # install Ruby deps (delete Gemfile.lock first if it errors)
bundle exec jekyll serve          # build + serve at localhost:4000, rebuild on change
bundle exec jekyll liveserve      # same, plus live-reload in the browser (via the hawkins gem)
bundle exec jekyll build          # one-off build into _site/ (gitignored)
```

- Local serving picks up `_config.dev.yml` **only if you pass it explicitly**:
  `bundle exec jekyll serve --config _config.yml,_config.dev.yml` (turns off analytics, sets `url` to localhost,
  expands SCSS instead of compressing).
- **`_config.yml` is not reloaded while the server runs** — restart `jekyll serve` after editing it. All other
  files hot-reload.
- JavaScript is hand-bundled, not part of the Jekyll build. After editing anything in `assets/js/`, regenerate
  the minified bundle: `npm run build:js` (concatenates + uglifies vendor/plugin/`_main.js` into
  `assets/js/main.min.js`, which is what the site actually loads). `npm run watch:js` does this on change.

## Deploy

Push to `master` → GitHub Pages rebuilds and publishes automatically. There is no separate `gh-pages` branch
build step and no Actions workflow. Because the plugin set is constrained to the GitHub Pages whitelist
(`whitelist:` / `plugins:` in `_config.yml`), any plugin not on that list will silently not run in production —
don't add gems expecting them to take effect.

## Content model

Everything is front-matter-driven Markdown/HTML. To add content you add a file to the right folder, not edit a
central index.

- **`_posts/`** — all articles, `YYYY-MM-DD-slug.md`, each with an explicit `permalink:` (never change existing
  permalinks). All posts list on the single `/blog/` page; categories/tags in front matter are optional
  metadata. Image/asset-heavy posts keep a companion directory of the same name next to the `.md` file — keep
  the two in sync when renaming or deleting. As of 2026-08-30 every past post sits in `_drafts/` awaiting the
  owner's rewrite; republish by moving a file back to `_posts/` (reuse the exact old permalink for 2023 posts).
- **`_pages/`** — standalone pages. `_pages/about.md` is the site homepage (`permalink: /`); its animated
  greeting is a small inline grapheme-aware typewriter (Intl.Segmenter — replaced typed.js, which split flag
  emoji into code units and made them render late). `blog.html` is the only internal nav destination; nav
  lives in `_data/navigation.yml` (Publications links out to Google Scholar).
- **Collections** (`_publications/`, `_portfolio/`, `_talks/`, `_teaching/`) are configured in `_config.yml`
  but intentionally empty — the academicpages placeholder content was removed in 2026. Don't add to them
  without also recreating listing pages.
- **`_drafts/`** — unpublished drafts (only shown with `--drafts`).
- Homelab/networking posts must never contain public IPs, DDNS hostnames, ISP names, or internal topology.
- Prose style for every post follows `docs/writing-guide.md` (plain descriptive titles and headings, no aphorism
  closers, no personified hardware, no em dashes). Run `python3 scripts/prose-tells.py _posts/<file>.md` before
  publishing to count leftover tells.

## Structure

- `_layouts/` — page templates (`single`, `splash`, `talk`, `archive`, `default`, taxonomy archives).
- `_includes/` — reusable Liquid partials assembled by layouts. Site-specific `<head>` additions
  (favicons, MathJax, academicons CSS) go in `_includes/head/custom.html`.
- `_data/` — `navigation.yml` (nav bar), `authors.yml` (multi-author bios), `ui-text.yml` (i18n theme strings),
  `comments/` (Staticman-submitted comments).
- `_sass/` + `assets/css/main.scss` — styles. `main.scss` is the single SCSS entry point that `@import`s
  everything in `_sass/`; Jekyll compiles it (compressed in prod, expanded in dev). Edit partials in `_sass/`,
  not compiled CSS.
- `markdown_generator/` — optional Python scripts / Jupyter notebooks that generate `_publications/` and
  `_talks/` Markdown from `.tsv` or BibTeX files. Not part of the site build; run manually when bulk-importing.
- `talkmap/` + `talkmap.py` / `talkmap.ipynb` — generates an interactive map of talk locations (opt-in via
  `talkmap_link` in `_config.yml`).

## Conventions

- Match the existing front-matter shape when adding to a collection — copy the fields from a sibling file rather
  than inventing keys. Per-collection defaults (layout, `author_profile`, `share`, `comments`) come from
  `defaults:` in `_config.yml`, so individual files only need to override.
- This is a detached fork; upstream `README.md`/`CONTRIBUTING.md` describe the generic template's PR conventions
  (issues tagged `code change`) and do **not** apply to this personal site.
