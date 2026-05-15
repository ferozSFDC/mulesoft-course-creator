"""Centralised client factory and feature flags for the Course Creation System."""

import json
import os
from pathlib import Path

import httpx
import anthropic

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


def validate_environment() -> None:
    """
    Validate all required environment variables at startup.
    Raises EnvironmentError listing every missing variable so the user
    can fix them all in one go rather than discovering them one by one.
    """
    missing = []

    if not (os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")):
        missing.append(
            "ANTHROPIC_AUTH_TOKEN (Salesforce internal token) or ANTHROPIC_API_KEY"
        )

    if not os.environ.get("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY (required for web search in the Researcher agent)")

    if missing:
        raise EnvironmentError(
            "Missing required environment variables:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nSee .env.example for the full list of required variables."
        )


def make_client() -> anthropic.Anthropic:
    """
    Return a configured Anthropic client.

    Validates credentials on first call. The caller is responsible for
    closing the underlying httpx.Client when done (use as a context manager
    or call client.close() explicitly).
    """
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")

    if not auth_token:
        raise EnvironmentError(
            "Set ANTHROPIC_AUTH_TOKEN (Salesforce internal token) "
            "or ANTHROPIC_API_KEY before running."
        )

    if base_url:
        transport = _BedrockProxyTransport(base_url, auth_token, _SSL_VERIFY)
        http_client = httpx.Client(transport=transport)
        return anthropic.Anthropic(api_key=auth_token, base_url=base_url, http_client=http_client)
    return anthropic.Anthropic(api_key=auth_token)


def thinking_param() -> dict | None:
    """
    Return the thinking + effort parameters for Opus 4.7, or None when
    thinking is disabled. effort="xhigh" is the recommended default for
    intelligence-sensitive work on Opus 4.7.
    """
    if not USE_THINKING:
        return None
    return {"type": "adaptive"}


def output_config_param() -> dict | None:
    """Return output_config with effort=xhigh when thinking is enabled."""
    if not USE_THINKING:
        return None
    return {"effort": "xhigh"}
