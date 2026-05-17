"""Flask entrypoint for the HTML/CSS dashboard migration."""

from __future__ import annotations

import os

from lesotho_property_ai.web.app import create_app

app = create_app()


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=False, use_reloader=False)


