import React from "react";
import ReactDOM from "react-dom";
import { App, AppConfigProvider } from "@councildataproject/cdp-frontend/dist/index.es.js";

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
        name: "Richmond (IN)",
        timeZone: "America/Indiana/Indianapolis",
        footerLinksSections: [],
    },
    features: {
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
