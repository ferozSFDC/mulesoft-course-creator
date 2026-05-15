"""Centralised client factory and feature flags for the Course Creation System."""

import json
import os
from pathlib import Path

import httpx
import anthropic

# ── Authentication ─────────────────────────────────────────────────────────────
_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")

if not _AUTH_TOKEN:
    raise EnvironmentError(
        "Set ANTHROPIC_AUTH_TOKEN (Salesforce internal token) "
        "or ANTHROPIC_API_KEY before running."
    )

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-7")

# ── Extended thinking ─────────────────────────────────────────────────────────
# Disabled by default because Bedrock proxies do not always support it.
# Set USE_THINKING=1 in your environment to enable adaptive thinking.
USE_THINKING = os.environ.get("USE_THINKING", "0") == "1"

# ── SSL ───────────────────────────────────────────────────────────────────────
# The Salesforce proxy uses a corporate TLS cert. We ship a combined CA bundle
# (certifi + macOS system keychain) so httpx can verify the connection properly.
_CA_BUNDLE = Path(__file__).parent / "sf-ca-bundle.pem"
_SSL_VERIFY: str | bool = str(_CA_BUNDLE) if _CA_BUNDLE.exists() else True


class _BedrockProxyTransport(httpx.BaseTransport):
    """
    Rewrites Anthropic SDK requests to match the Salesforce Bedrock proxy format:
      - URL:  {base}/model/{model}/invoke  (model extracted from JSON body)
      - Body: drops "model" key, adds "anthropic_version": "bedrock-2023-05-31"
      - Auth: replaces x-api-key with Authorization: Bearer
    """

    def __init__(self, base_url: str, token: str, verify: str | bool) -> None:
        self._base = base_url
        self._token = token
        self._inner = httpx.HTTPTransport(verify=verify)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/v1/messages"):
            body = json.loads(request.content)
            model = body.pop("model", MODEL)
            body["anthropic_version"] = "bedrock-2023-05-31"
            new_request = httpx.Request(
                method="POST",
                url=f"{self._base}/model/{model}/invoke",
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                content=json.dumps(body).encode(),
            )
            return self._inner.handle_request(new_request)
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


def make_client() -> anthropic.Anthropic:
    """Return a configured Anthropic client pointing at the Salesforce proxy."""
    if _BASE_URL:
        transport = _BedrockProxyTransport(_BASE_URL, _AUTH_TOKEN, _SSL_VERIFY)
        http_client = httpx.Client(transport=transport)
        # base_url must be set to something; the transport rewrites the final URL anyway.
        return anthropic.Anthropic(api_key=_AUTH_TOKEN, base_url=_BASE_URL, http_client=http_client)
    return anthropic.Anthropic(api_key=_AUTH_TOKEN)


def thinking_param() -> dict | None:
    """Return the thinking parameter dict, or None when thinking is disabled."""
    return {"type": "adaptive"} if USE_THINKING else None
