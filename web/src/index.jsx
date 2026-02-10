import React from "react";
import ReactDOM from "react-dom";
import { App, AppConfigProvider } from "./cdp_fork/index.es.js";

import "@councildataproject/cdp-frontend/dist/index.css";

const config = {
    firebaseConfig: {
        options: {
            projectId: "cdp-richmond-in-jvrzndvq",
            storageBucket: "cdp-richmond-in-jvrzndvq.appspot.com",
        },
        settings: {},
    },
    municipality: {
        name: "Richmond & Wayne County, Indiana",
        timeZone: "America/Indiana/Indianapolis",
        footerLinksSections: [],
    },
    features: {
        // cdp-richmond-indiana: Enable single-page Events layout mode (suppresses CDP multi-tab nav/routes via patch-package).
        singlePageEvents: true,
        // enableClipping: true,
    },
}

ReactDOM.render(
    <div>
        <AppConfigProvider appConfig={config}>
            <App />
        </AppConfigProvider>
    </div>,
    document.getElementById("root")
);
