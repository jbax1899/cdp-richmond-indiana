# Operations Reference

This document is maintainer-focused and complements `README.md`.

## Core Workflows

- Infrastructure deploy: `.github/workflows/deploy-infra.yml`
- Event gather pipeline: `.github/workflows/event-gather-pipeline.yml`
- Event index pipeline: `.github/workflows/event-index-pipeline.yml`
- Special event ingest: `.github/workflows/process-special-event.yml`
- Ad-hoc runner commands: `.github/workflows/run-script.yml`
- Web deploy: `.github/workflows/deploy-web.yml`
- Main branch build checks: `.github/workflows/build-main.yml`
- Pull request checks: `.github/workflows/check-pr.yml`

## Run Cadence

- Event gather is scheduled daily at `00:00 UTC`.
- Event index is scheduled weekly on Thursdays at `03:26 UTC`.
- Infrastructure and web deploy run on `main` pushes and weekly schedule.

## Typical Manual Actions

### Reprocess a time window

- Use Event Gather workflow dispatch with `from` and `to`.
- Reference: `admin-docs/manual-event-gather.md`

### Ingest a custom/special event

- Use Process Special Event workflow dispatch with JSON payload.
- Reference: `admin-docs/manual-event-gather.md`

### Run one-off script in CI

- Use Run Command workflow dispatch.
- Pass command/args exactly as they should run from `python/`.
- Reference: `admin-docs/running-extra-scripts.md`

## Notes

- Event gather and special event workflows patch Whisper transcription language to English and apply repo-specific thumbnail patching before running CDP backend entrypoints.
- Index generation/upload is global and can be expensive; run during low-traffic windows when possible.
- Setup/bootstrap guidance is in `SETUP/README.md`.
