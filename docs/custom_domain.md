# Custom Domain Runbook

This document lists every place to update when changing the web domain for this project.

Current custom host:

- `richmondmeetings.jordanmakes.dev`

Future target (example):

- root/apex domain such as `jordanmakes.dev`

## Repository Updates

Update these files before deploying web:

1. `web/package.json`
   - Update `homepage`.
   - Set `homepage` to the active web host root (example: `https://richmondmeetings.jordanmakes.dev`).
2. `web/public/index.html`
   - Replace hardcoded domain references in:
   - `favicon` URL
   - `manifest` URL
   - `og:url`
   - `og:image`
   - `twitter:url`
   - `twitter:image`
   - Plausible `data-domain`
   - Prefer same-origin relative paths for local assets (`/favicon.ico`, `/manifest.json`, `/wayne-county-courthouse-social.jpg`).
3. `web/public/CNAME`
   - Ensure this file exists and contains exactly the active custom host.
   - Example content for current setup: `richmondmeetings.jordanmakes.dev`
4. `web/public/404.html` and `web/public/index.html` SPA redirect bootstrap
   - Keep GitHub Pages SPA fallback redirect in place when using clean URLs (BrowserRouter / non-hash routing).
   - This preserves deep-link refresh behavior on static hosting.
5. `README.md`
   - Update "Live site" URL.
6. `cookiecutter.yaml`
   - Update `hosting_web_app_address` so template metadata stays accurate.
7. `SETUP/README.md` and `admin-docs/README_old.md` (if retained)
   - Update any example/live-site links that still reference prior domains.

## GitHub Pages Settings

In repository settings:

1. Go to `Settings -> Pages`.
2. Set `Custom domain` to the exact host in `web/public/CNAME`.
3. Wait for certificate issuance.
4. Enable `Enforce HTTPS` once available.

Note: keep `web/public/CNAME` aligned with Pages settings so deploys do not drift domain config.

## DNS Provider Settings

### Subdomain setup (current pattern)

For host like `richmondmeetings.jordanmakes.dev`:

1. Create `CNAME` record:
   - Name: `richmondmeetings`
   - Target: `<github-user>.github.io` (currently `jbax1899.github.io`)
2. If your DNS provider has optional proxy/CDN mode, start with direct DNS until GitHub HTTPS is active.
3. Optionally enable proxy/CDN mode after HTTPS is stable.

### Root/apex domain setup (future pattern)

For host like `jordanmakes.dev`:

1. Follow current GitHub Pages apex DNS guidance in official docs.
2. Configure apex records in your DNS provider for GitHub Pages (record type depends on provider support, e.g. A/ALIAS/ANAME).
3. Keep `www` behavior explicit (redirect or separate host), do not leave ambiguous.

### Optional provider-specific note (Cloudflare)

- If using Cloudflare, proxy mode can be re-enabled after GitHub certificate and `Enforce HTTPS` are confirmed.

## Deploy and Verify

After updates:

1. Deploy web (`.github/workflows/deploy-web.yml`).
2. Purge DNS/CDN cache if your provider uses edge caching or proxying.
3. Hard-refresh browser.
4. Verify:
   - Page returns `200`.
   - JS/CSS assets load from valid paths and return `200`.
   - No requests to old host/path.
   - Manifest loads from same origin (no CORS error).
   - `Enforce HTTPS` remains enabled in GitHub Pages.

## Known Non-Blocking Console Noise

- `plausible.hash.js` blocked by client often means ad blocker/privacy extension and is not a site outage by itself.
