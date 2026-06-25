"""
Local-app security primitives:
  - a per-install bearer token (defense-in-depth for the local API)
  - path-confinement helpers so served files can never escape known photos

The token is enforced only when PV_REQUIRE_AUTH=1 (set by serve.py for real
runs). Tests and library imports leave it off so behavior is unchanged.
"""
import os
import secrets
import stat

from constants import DATA_DIR

TOKEN_PATH = os.path.join(DATA_DIR, ".auth_token")

# /api paths reachable without a token even when auth is enabled:
#  - health: harmless probe
#  - token: how the dev SPA (served by Vite, no HTML injection) bootstraps
EXEMPT_API_PATHS = {"/api/health", "/api/token"}

_token_cache: str | None = None


def get_token() -> str:
    """Read the persisted token, generating + persisting one on first use."""
    global _token_cache
    if _token_cache:
        return _token_cache
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(TOKEN_PATH):
        try:
            with open(TOKEN_PATH) as f:
                tok = f.read().strip()
            if tok:
                _token_cache = tok
                return tok
        except Exception:
            pass
    tok = secrets.token_urlsafe(32)
    try:
        with open(TOKEN_PATH, "w") as f:
            f.write(tok)
        os.chmod(TOKEN_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 0600 (best-effort on Windows)
    except Exception as e:  # pragma: no cover
        print(f"[security] could not persist auth token: {e}")
    _token_cache = tok
    return tok


def auth_enabled() -> bool:
    return os.environ.get("PV_REQUIRE_AUTH", "0") == "1"


def token_matches(value: str | None) -> bool:
    """Constant-time compare of a raw token value against the install token."""
    return bool(value) and secrets.compare_digest(value, get_token())


def check_bearer(authorization_header: str | None) -> bool:
    """True if the Authorization header carries the correct bearer token."""
    if not authorization_header:
        return False
    parts = authorization_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False
    return token_matches(parts[1].strip())


def request_authorized(authorization_header: str | None, query_token: str | None) -> bool:
    """
    Accept the token from either the Authorization header (fetch/XHR) or a
    `_t` query param (used by <img> tags, which can't set headers).
    """
    return check_bearer(authorization_header) or token_matches(query_token)


def is_safe_real_path(path: str) -> str | None:
    """
    Resolve symlinks/.. and return the canonical path if it points to an
    existing regular file, else None. Use before serving any file to a client.
    """
    try:
        rp = os.path.realpath(os.path.abspath(path))
        if os.path.isfile(rp):
            return rp
    except Exception:
        pass
    return None
