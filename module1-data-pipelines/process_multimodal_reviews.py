"""
M1L2 — Process Multimodal Data with LLMs
========================================
Loops over the user-review dataset, sends each recipe/food image to a
vision-capable LLM (Llama on watsonx.ai) to generate a caption, assigns the
caption to the review record, and saves augmented_user_review.json.

Reconstructed to the lab's screenshot and grading requirements
(M1L2_caption_all_recipes: the for loop that calls the vision LLM and
assigns the generated captions to each entry in the JSON dataset).

Credentials via environment variables only:
    WATSONX_APIKEY, WATSONX_URL (default us-south), WATSONX_PROJECT_ID
"""

import base64
import json
import os
from pathlib import Path

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

REVIEW_INPUT_PATH = Path("user_review.json")
OUTPUT_PATH = Path("augmented_user_review.json")

WATSONX_URL = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_APIKEY = os.environ.get("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "skills-network")
VISION_MODEL_ID = os.environ.get(
    "WATSONX_VISION_MODEL", "meta-llama/llama-3-2-11b-vision-instruct"
)

CAPTION_PROMPT = (
    "Describe this food image in one detailed sentence: name the dish if "
    "recognizable, its key visible ingredients, and its presentation style. "
    "Return only the caption sentence."
)


def encode_image(image_path: str) -> str:
    """Read an image file and return its base64 string for the vision model."""
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def vision_llm_caption(image_b64: str, prompt: str = CAPTION_PROMPT) -> str:
    """Send one image + prompt to the watsonx.ai vision model, return caption."""
    if WATSONX_APIKEY:
        credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_APIKEY)
    else:
        credentials = Credentials(url=WATSONX_URL)
    model = ModelInference(
        model_id=VISION_MODEL_ID,
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID,
        params={"temperature": 0.2, "max_tokens": 200},
    )
    response = model.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ]
    )
    return response["choices"][0]["message"]["content"].strip()


def caption_all_recipes(user_review_data: list[dict]) -> list[dict]:
    """The M1L2 core loop: for each review with an image, call the vision LLM
    and assign the generated caption to the record."""
    for i, review in enumerate(user_review_data):
        image_path = review.get("image_path")
        if not image_path or not Path(image_path).exists():
            review["image_description"] = "N/A"
            print(f"Review {i}: no image found, marked N/A")
            continue
        image_b64 = encode_image(image_path)
        caption = vision_llm_caption(image_b64)
        review["image_description"] = caption
        print(f"Review {i}: captioned — {caption[:60]}...")
    return user_review_data


def save_augmented_reviews(
    user_review_data: list[dict], filename: Path = OUTPUT_PATH
) -> None:
    """Save — verbatim per the lab's final save cell."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(user_review_data, f, indent=4)


if __name__ == "__main__":
    with open(REVIEW_INPUT_PATH, "r", encoding="utf-8") as f:
        user_review_data = json.load(f)
    user_review_data = caption_all_recipes(user_review_data)
    save_augmented_reviews(user_review_data)
    print(f"Saved {len(user_review_data)} augmented reviews to {OUTPUT_PATH}")
