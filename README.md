# Richmond & Wayne County Public Meetings (Indiana)

A searchable portal for Richmond and Wayne County public meetings, with videos and machine transcripts, built with Council Data Project (CDP).

This is a community-run project, not an official municipal record, and it links back to the original public sources.

Live site: https://richmondmeetings.jordanmakes.dev

What you can do on the live site:
- Browse meetings by body and date
- Search inside transcripts
- Jump from search results to source video and document links

## Quick Start (3 minutes)

Prerequisites:
- Python `3.10`
- Node.js `16.x` (required by current web build/workflows)
- `git`

If you only want to work on the frontend:
```powershell
cd web
npm install
npm start
```

If you want the full local dev setup:
```powershell
cd python
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install .[test]
```
If activation fails in PowerShell, run:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

```powershell
cd web
npm install
npm start
```

## What This Repo Is Built On

- [Council Data Project (CDP)](https://councildataproject.org/) for civic meeting ingestion, indexing, and UI foundations
- `cdp-backend[pipeline]==4.1.3` (`python/setup.py`)
- `@councildataproject/cdp-frontend@3.3.0` with an instance-specific maintained fork (`web/src/cdp_fork/index.es.js`)

## Instance-Specific Changes

### Backend (instance-specific)
- Adds a custom Internet Archive scraper in `python/cdp_richmond_(in)_backend/scraper.py`
- Improves body mapping and media ranking for Richmond/Wayne meetings
- Resolves redirected media URLs and reports ingest skip summaries
- Patches thumbnail generation in `python/cdp_patches.py` to avoid blank lead-in frames
- Enforces English transcription and repo-specific thumbnail patching in gather/special-event workflows

### Frontend (instance-specific)
- Maintained instance-specific fork: `web/src/cdp_fork/index.es.js`
- Robust download URL handling for `http(s)` and `gs://`
- Single-page Events mode via `features.singlePageEvents`
- Instance-specific hero/footer components:
  - `web/src/components/LocalHero.jsx`
  - `web/src/components/LocalFooter.jsx`
- `display_keywords` support for event cards/search context

### Tooling
- `python/regenerate_data.py` for targeted delete/regenerate of transcripts, thumbnails, and index docs
- `python/generate_display_keywords.py` for optional LLM-based keyword generation

### CI / Ops
- Core deployment/index/gather workflows are in `.github/workflows/`
- Maintainers are the people operating the hosted instance (pipelines, infra, and deployments)
- Short maintainer view is below; full workflow reference is in `docs/operations.md`

## Optional Maintainer Scripts

Credentials are resolved by repo scripts in this order:
1. `--credentials-file`
2. `GOOGLE_APPLICATION_CREDENTIALS`
3. `python/event-gather-config.json`
4. `.keys/*.json` auto-detection

### Regenerate artifacts (`regenerate_data.py`)

Dry-run single event:
```powershell
cd python
python regenerate_data.py --event-id <EVENT_DOC_ID> --delete thumbnails --regenerate thumbnails
```

Apply transcript regeneration:
```powershell
cd python
python regenerate_data.py --event-id <EVENT_DOC_ID> --delete transcripts --regenerate transcripts --apply
```

Global index regeneration:
```powershell
cd python
python regenerate_data.py --event-id <EVENT_DOC_ID> --regenerate index --index-parallel --apply
```

### LLM-based keyword generation (`generate_display_keywords.py`)

Keyword generation is optional and not required for core event indexing/search.

Single event:
```powershell
cd python
$env:GEMINI_API_KEY = "<your_api_key>"
python generate_display_keywords.py --event-id <EVENT_DOC_ID> --changed-only --apply
```

All events:
```powershell
cd python
$env:GEMINI_API_KEY = "<your_api_key>"
python generate_display_keywords.py --all --changed-only --apply
```

## Ops Summary (Maintainers)

- Infrastructure deploy: `.github/workflows/deploy-infra.yml`
- Event gather: `.github/workflows/event-gather-pipeline.yml` (daily at `00:00 UTC` + manual range runs)
- Event index: `.github/workflows/event-index-pipeline.yml` (Thursdays at `03:26 UTC` + manual runs)
- Full workflow inventory and run guidance: `docs/operations.md`

## Project Boundaries & Disclaimers

- This project is independent and unofficial.
- It does not replace official records, minutes, or legal notices.
- It does not host the original meeting media; source publishers control availability and licensing.
- Most transcripts are machine-generated and may contain errors.
- Corrections/feedback are welcome via issues: https://github.com/jbax1899/cdp-richmond-indiana/issues
- Source code license: Mozilla Public License 2.0 (MPL-2.0) (`LICENSE`).

## More Documentation

- Instance admin docs: `admin-docs/`
- Setup history/bootstrap docs: `SETUP/README.md`
- Workflow operations reference: `docs/operations.md`
