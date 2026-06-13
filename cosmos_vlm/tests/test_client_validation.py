from __future__ import annotations

import pytest

from cosmos_vlm.predict25.client import submit_image_to_video


def test_submit_rejects_non_url_image() -> None:
    with pytest.raises(ValueError):
        submit_image_to_video(
            base_url="https://api.wavespeed.ai/api/v3",
            api_key="dummy",
            prompt="hello",
            image="/tmp/local.png",
        )

