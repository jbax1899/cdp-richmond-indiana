import React from "react";

export default function LocalFooter() {
  return (
    <footer
      className="mzp-c-footer cdp-local-footer"
      style={{
        marginTop: 48,
        paddingTop: 40,
        paddingBottom: 40,
        background:
          "linear-gradient(180deg, rgba(246, 248, 250, 0) 0%, rgba(246, 248, 250, 1) 22%, rgba(246, 248, 250, 1) 100%)",
        borderTop: "1px solid rgba(0,0,0,0.08)",
      }}
    >
      <div className="mzp-l-content">
        <div style={{ maxWidth: 980, margin: "0 auto" }}>
          <h2 className="mzp-u-title-sm" style={{ marginTop: 0, marginBottom: 16 }}>
            About this project
          </h2>

          <p style={{ marginTop: 0, marginBottom: 12, fontSize: "1.05em" }}>
            This is a public tool for browsing recent local government meetings in Richmond and Wayne County, Indiana. It
            is powered by the open-source{" "}
            <a href="https://councildataproject.org" target="_blank" rel="noopener noreferrer external">
              Council Data Project
            </a>{" "}
            and maintained independently.
          </p>

          <p style={{ marginTop: 0, marginBottom: 12 }}>
            View the source code or contribute on{" "}
            <a href="https://github.com/jbax1899/cdp-richmond-indiana" target="_blank" rel="noopener noreferrer external">
              GitHub
            </a>
            .
          </p>

          <p style={{ marginTop: 0, marginBottom: 20 }}>
            Have feedback or corrections?{" "}
            <a
              href="https://github.com/jbax1899/cdp-richmond-indiana/issues"
              target="_blank"
              rel="noopener noreferrer external"
            >
              Open an issue on GitHub
            </a>
            .
          </p>

          <hr
            style={{
              border: 0,
              borderTop: "1px solid rgba(0,0,0,0.10)",
              marginTop: 24,
              marginBottom: 24,
            }}
          />

          <div style={{ fontSize: "0.95em", opacity: 0.9, lineHeight: 1.55 }}>
            <p style={{ marginTop: 0, marginBottom: 10 }}>Transcripts are auto-generated and may contain errors.</p>
            <p style={{ marginTop: 0, marginBottom: 10 }}>Content © individual contributors.</p>

            <p style={{ marginTop: 0, marginBottom: 10 }}>
              Site source code licensed under{" "}
              <a href="https://opensource.org/licenses/MIT" target="_blank" rel="noopener noreferrer external">
                MIT
              </a>
              .
            </p>

            <p style={{ marginTop: 0, marginBottom: 10 }}>
              Meeting media and transcripts are shared under a{" "}
              <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener noreferrer external">
                Creative Commons Attribution 4.0
              </a>{" "}
              license.
            </p>

            <p style={{ marginTop: 0, marginBottom: 10 }}>
              Styled using{" "}
              <a href="https://protocol.mozilla.org" target="_blank" rel="noopener noreferrer external">
                Mozilla Protocol
              </a>
              . Artwork by{" "}
              <a href="https://undraw.co" target="_blank" rel="noopener noreferrer external">
                unDraw
              </a>
              .
            </p>

            <p style={{ marginTop: 16, marginBottom: 0 }}>
              Hero image:{" "}
              <a
                href="https://commons.wikimedia.org/wiki/File:WayneCountyCourthouse.jpg"
                target="_blank"
                rel="noopener noreferrer external"
              >
                WayneCountyCourthouse.jpg
              </a>{" "}
              by{" "}
              <a href="https://commons.wikimedia.org/wiki/User:Greg5030" target="_blank" rel="noopener noreferrer external">
                Greg Hume
              </a>
              , licensed under{" "}
              <a href="https://creativecommons.org/licenses/by-sa/3.0/" target="_blank" rel="noopener noreferrer external">
                CC BY-SA 3.0
              </a>
              .
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}

