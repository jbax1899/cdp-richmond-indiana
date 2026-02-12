import React from "react";
import { Link, useLocation } from "react-router-dom";

function getBreadcrumbs(pathname) {
  const crumbs = [{ to: "/", label: "Home" }];
  const segments = String(pathname || "/")
    .split("/")
    .filter(Boolean);

  if (segments.length === 0) {
    return crumbs;
  }

  const [s0, s1] = segments;

  if (s0 === "events") {
    crumbs.push({ to: "/events", label: "Events" });
    if (s1 === "search") {
      crumbs.push({ to: "/events/search", label: "Search" });
    } else if (s1) {
      // We can't know the event title here (hero is outside the <Route/>), so keep it generic.
      crumbs.push({ to: `/events/${s1}`, label: "Meeting" });
    }
    return crumbs;
  }

  if (s0 === "people") {
    crumbs.push({ to: "/people", label: "People" });
    if (s1) {
      crumbs.push({ to: `/people/${s1}`, label: "Person" });
    }
    return crumbs;
  }

  if (s0 === "matters") {
    crumbs.push({ to: "/matters", label: "Matters" });
    if (s1) {
      crumbs.push({ to: `/matters/${s1}`, label: "Matter" });
    }
    return crumbs;
  }

  // Fallback: show a short, readable crumb for the first segment only.
  const label = decodeURIComponent(s0).replace(/[-_]+/g, " ");
  crumbs.push({ to: `/${s0}`, label: label.charAt(0).toUpperCase() + label.slice(1) });
  return crumbs;
}

export default function LocalHero({ municipalityName }) {
  const publicUrl = (typeof process !== "undefined" && process.env && process.env.PUBLIC_URL) ? process.env.PUBLIC_URL : "";
  const heroImageUrl = `${publicUrl}/wayne-county-courthouse-hero.webp`;
  const location = useLocation();
  const breadcrumbs = getBreadcrumbs(location && location.pathname);
  const showBreadcrumbBar = true;

  return (
    <React.Fragment>
      <section
        className="cdp-local-hero"
        style={{
          position: "relative",
          backgroundColor: "#0b1b26",
          backgroundImage: `url(${heroImageUrl})`,
          backgroundSize: "cover",
          backgroundPosition: "center",
        }}
      >
        <div
          aria-hidden="true"
          style={{
            position: "absolute",
            inset: 0,
            background:
              "linear-gradient(180deg, rgba(11,27,38,0.72) 0%, rgba(11,27,38,0.84) 55%, rgba(11,27,38,0.92) 100%)",
          }}
        />

        <div
          className="mzp-l-content"
          style={{
            position: "relative",
            paddingTop: 56,
            paddingBottom: 56,
          }}
        >
          <div style={{ maxWidth: 980 }}>
            <p
              className="mzp-u-title-2xs"
              style={{
                color: "rgba(255,255,255,0.85)",
                marginTop: 0,
                marginBottom: 12,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
              }}
            >
              {municipalityName}
            </p>

            <h1
              className="mzp-u-title-lg"
              style={{
                color: "white",
                marginTop: 0,
                marginBottom: 12,
                textShadow: "0 1px 18px rgba(0,0,0,0.35)",
              }}
            >
              Public Meetings
            </h1>

            <p
              className="mzp-u-body-lg"
              style={{
                color: "rgba(255,255,255,0.88)",
                margin: 0,
                maxWidth: 760,
              }}
            >
              Search recent meetings and transcripts.
            </p>

            <p
              style={{
                marginTop: 28,
                marginBottom: 0,
                fontSize: 12,
                lineHeight: 1.4,
                color: "rgba(255,255,255,0.70)",
                maxWidth: 760,
              }}
            >
              Unofficial community project • Transcripts are auto-generated and may contain errors.
            </p>
          </div>
        </div>
      </section>

      {showBreadcrumbBar ? (
        <div
          style={{
            borderBottom: "1px solid rgba(0,0,0,0.08)",
            background: "#f9f9fb",
          }}
        >
          <div className="mzp-l-content" style={{ paddingTop: 12, paddingBottom: 12 }}>
            <div style={{ maxWidth: 980 }}>
              <nav
                aria-label="Breadcrumb"
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 10,
                  alignItems: "center",
                }}
              >
                {breadcrumbs.map((c, idx) => {
                  const isLast = idx === breadcrumbs.length - 1;
                  return (
                    <React.Fragment key={`${c.to}:${c.label}`}>
                      {idx > 0 ? (
                        <span style={{ color: "rgba(0,0,0,0.35)" }} aria-hidden="true">
                          /
                        </span>
                      ) : null}
                      {isLast ? (
                        <span style={{ color: "rgba(0,0,0,0.78)", fontSize: 14 }}>
                          {c.label}
                        </span>
                      ) : (
                        <Link
                          to={c.to}
                          style={{
                            color: "rgba(0,0,0,0.82)",
                            fontSize: 14,
                            textDecoration: "none",
                            borderBottom: "1px solid rgba(0,0,0,0.25)",
                            paddingBottom: 1,
                          }}
                        >
                          {c.label}
                        </Link>
                      )}
                    </React.Fragment>
                  );
                })}
              </nav>
            </div>
          </div>
        </div>
      ) : null}
    </React.Fragment>
  );
}
