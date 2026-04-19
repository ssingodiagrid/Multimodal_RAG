import os
from pathlib import Path
from typing import Any, Optional

import vertexai
from vertexai.generative_models import GenerativeModel, Part


def _ensure_google_application_credentials() -> None:
    """Set GOOGLE_APPLICATION_CREDENTIALS for Vertex if not already set."""
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        p = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).expanduser()
        if p.is_file():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p.resolve())
        return

    explicit = os.getenv("GCP_SERVICE_ACCOUNT_KEY_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path.resolve())
            return

    creds_dir = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_DIR")
    fname = os.getenv("GCP_KEY_FILENAME", "gcp-key.json")
    if creds_dir:
        path = Path(creds_dir).expanduser() / fname
        if path.is_file():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path.resolve())
            return

    root_key = (Path.cwd() / "gcp-key.json").resolve()
    if root_key.is_file():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(root_key)


class GeminiClient:
    """Gemini via Vertex AI (service account), not Google AI Studio API key."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        _ensure_google_application_credentials()
        proj = project or os.getenv("GCP_PROJECT_ID")
        loc = location or os.getenv("GCP_LOCATION", "us-central1")
        if not proj:
            raise ValueError(
                "GCP_PROJECT_ID is required for Vertex AI. Set it in the environment or .env."
            )
        name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
        vertexai.init(project=proj, location=loc)
        self.model = GenerativeModel(name)

    @staticmethod
    def _text_from_response(resp: Any) -> str:
        if hasattr(resp, "text") and resp.text:
            return resp.text
        if getattr(resp, "candidates", None):
            parts: list[str] = []
            for c in resp.candidates:
                content = getattr(c, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        t = getattr(part, "text", None)
                        if t:
                            parts.append(t)
            if parts:
                return "".join(parts)
        return str(resp)

    def answer(self, prompt: str) -> str:
        resp: Any = self.model.generate_content(prompt)
        return self._text_from_response(resp)

    def answer_with_image(
        self, text_prompt: str, image_bytes: bytes, mime_type: str = "image/png"
    ) -> str:
        """Multimodal: text instructions + one inline image (Vertex Part.from_data)."""
        if not image_bytes:
            return self.answer(text_prompt)
        mt = mime_type if mime_type.startswith("image/") else "image/png"
        resp: Any = self.model.generate_content(
            [text_prompt, Part.from_data(data=image_bytes, mime_type=mt)]
        )
        return self._text_from_response(resp)

    def answer_with_images(
        self,
        text_prompt: str,
        images: list[tuple[bytes, str]],
    ) -> str:
        """Multimodal: one text block then multiple inline images (ColPali page evidence)."""
        if not images:
            return self.answer(text_prompt)
        parts: list[Any] = [text_prompt]
        for data, mime in images:
            mt = mime if mime.startswith("image/") else "image/png"
            parts.append(Part.from_data(data=data, mime_type=mt))
        resp: Any = self.model.generate_content(parts)
        return self._text_from_response(resp)
