# SEO Metadata Guidance

Keep this repo's public metadata consistent with the site branding and domain.

## Core Rules

- Keep title/description consistent across primary, Open Graph, and Twitter tags in `web/public/index.html`.
- Keep `og:url`, `twitter:url`, and canonical URL aligned with `cookiecutter.yaml` `hosting_web_app_address`.
- Keep app naming in `web/public/manifest.json` aligned with branding used in `README.md`.

## Social Preview Images

- Build social preview images from the original source image:
  - `web/public/wayne-county-courthouse.jpg`
- Do not use the trimmed hero image as the source:
  - `web/public/wayne-county-courthouse-hero.webp`
- Publish a web-optimized social image at:
  - `web/public/wayne-county-courthouse-social.jpg`
- Use `1200x630` output for broad social card compatibility.

## Quick Verification

1. View page source and confirm title/description are synchronized across primary, OG, and Twitter tags.
2. Confirm canonical URL and social URLs match the active domain.
3. Validate link previews after deploy and clear debugger caches as needed.
4. Re-check this after any domain update (see `docs/custom_domain.md`).
