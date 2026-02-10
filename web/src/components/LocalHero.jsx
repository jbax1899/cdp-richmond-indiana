import React from "react";

export default function LocalHero({ municipalityName }) {
  const publicUrl = (typeof process !== "undefined" && process.env && process.env.PUBLIC_URL) ? process.env.PUBLIC_URL : "";
  const heroImageUrl = `${publicUrl}/wayne-county-courthouse.jpg`;

  return (
    <section
      className="cdp-local-hero"
      style={{
        position: "relative",
        backgroundColor: "#0b1b26",
        backgroundImage: `url(${heroImageUrl})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
        borderBottom: "1px solid rgba(0,0,0,0.08)",
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
            Browse recent meeting videos and auto-generated transcripts.
          </p>
        </div>
      </div>
    </section>
  );
}

