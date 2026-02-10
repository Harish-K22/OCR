"""Mistral OCR via Pixtral vision model API."""

import base64
import requests
from pathlib import Path

from .base import OCRModel
import config


class MistralOCR(OCRModel):

    def load_model(self) -> None:
        if not config.MISTRAL_API_KEY:
            raise ValueError(
                "MISTRAL_API_KEY not set. Add it to your .env file."
            )

    def extract_text(self, image_path: str) -> str:
        img_b64 = self._encode_image(image_path)
        ext = Path(image_path).suffix.lstrip(".").lower()
        mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"

        payload = {
            "model": config.MISTRAL_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{img_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract ALL text from this image exactly as written. "
                                "Return only the extracted text, nothing else."
                            ),
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.MISTRAL_API_KEY}",
        }
        resp = requests.post(
            config.MISTRAL_API_URL, json=payload, headers=headers, timeout=60
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def get_name(self) -> str:
        return "Mistral OCR"

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
