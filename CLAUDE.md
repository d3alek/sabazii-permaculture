# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **not a software project**. It is a permaculture design specification repository for **Eco-village Sabaziy** (Еко-селище Сабазий), located near Belintash in the Rhodope Mountains, Bulgaria.

**Designers**: Danail, Tsvetelina, Aleksandar — team of 3 adults + 1 child.
**Client**: Yordanka & Dimitar Ivanovi (owners), Diana (on-site coordinator, diana@conscious.world). Company: Conscious EOOD.

## What This Repository Contains

- **final-design.md** — Pandoc-generated markdown from the Google Doc (source of truth). Used for local reference by Claude. Synced one-way: GDocs → local.
- **sync-gdoc.sh** — Script to sync Google Doc → local markdown (DOCX export + pandoc).
- **CONTEXT.md** — Master project brief synthesized from the client email and brainstorming sessions.
- **Maps** (JPG) — Contour, slope, aspect, property boundary, and overview maps from geodesic survey data. `ideas_map.jpg` is a brainstorm overlay.
- **Soil analysis PDF** (Bulgarian) — Lab results for plot 18509.
- **Site assessment questionnaire** (DOCX, Bulgarian) — Completed client questionnaire for the Belintash site.
- **"The Intelligent Gardener" PDF** — Reference book for soil fertility and mineral balancing (Steve Solomon).
- **sections/** — DEPRECATED. Old per-section markdown files. No longer maintained; use `final-design.md` instead.

## Sync Workflow: Google Docs ↔ GitHub

**Google Doc is the source of truth.** All editing and commenting happens there.

### Google Docs → Local / GitHub (automated)

**Always use the sync script** — never read the Google Doc via MCP (`get_doc_content`) for review or analysis. The MCP output loses formatting structure, while the pandoc-generated markdown preserves heading hierarchy, list nesting, and other structural cues that are essential for spotting formatting problems.

```bash
bash sync-gdoc.sh   # downloads DOCX, converts to GFM markdown
```

Then read and analyze `final-design.md` locally.

### Local → Google Docs (manual, preserves comments)

When Claude suggests content changes:
1. Claude describes the change and specifies the exact location in the document
2. The user applies the change manually in the Google Docs UI
3. This preserves all comments, formatting, and revision history
4. After changes, re-run the sync to update the local markdown

**Never write directly to the Google Doc via API for content changes** — this risks breaking comment anchors and revision history.

## GitHub Repository

- **Repo**: https://github.com/d3alek/sabazii-permaculture
- **Design document**: `content/final-design.md` (pandoc-generated, one-way sync from GDocs)

## Core Design Methodology: Water-Access-Structures (W-A-S)

The project follows a strict three-pillar sequence:

1. **Water** — Optimal water retention along slopes (swales, terraces, water storage). Determined algorithmically from contour/slope data.
2. **Access** — Roads and paths follow contour lines, placed *after* water elements are defined.
3. **Structures** — Buildings and installations placed in the spaces defined by water and access.

This ordering is non-negotiable in the design philosophy. W-A-S maps for the full ~200 dekar property come first; detailed planting design comes second.

## Two-Level Design Approach

1. **Big picture**: W-A-S design for the entire ~200 dekar property
2. **"Dream garden" (мечтаната градинка)**: A vegetable garden and a few fruit trees on ~0.5 dekar that the investors want to plant THIS YEAR, before earthworks (terracing, gabions, swales) are completed. The goal is a first harvest and visible result in the current season. The location must fit within the larger W-A-S design so nothing needs to be moved later.

## Key External Resources

- **Geodesic platform**: https://lcc-viewer.xgrids.com/pub/dbtbdp-sabazii (password: sabazii)
- **Google Doc (FDE-structured, active)**: https://docs.google.com/document/d/1WO1MAfWaJisfWueIRMQ0X0OPMheXIu2b2l3al_QKhu4/edit (ID: `1WO1MAfWaJisfWueIRMQ0X0OPMheXIu2b2l3al_QKhu4`)

## Google Workspace MCP

A `google-workspace` MCP server is configured for reading/writing Google Docs, Drive, and Sheets. User email: `akodzhabashev@gmail.com`.

### When to use MCP tools

- **DO NOT use `get_doc_content` for reading/reviewing** — use `sync-gdoc.sh` + read `final-design.md` instead. The MCP plain-text output strips heading levels, list structure, and formatting, making it impossible to catch formatting issues.
- **Small targeted edits only**: `find_and_replace_doc` for precise text replacements (preserves comments on unchanged text)
- **DO NOT use for bulk content changes** — instruct the user to edit in the Google Docs UI instead
- **`inspect_doc_structure(detailed=true)`** — only for finding insertion indices when programmatic edits are needed

### Auth

If OAuth token expires, write the auth URL to `auth_url.txt` and open with `xdg-open "$(cat auth_url.txt)"`.

## Language

All client-facing documents and most project files are in **Bulgarian**. The design methodology terminology (Water-Access-Structures, swales, food forest, etc.) uses English permaculture terms alongside Bulgarian.
