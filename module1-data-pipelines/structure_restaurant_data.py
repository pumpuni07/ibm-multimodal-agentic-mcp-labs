"""
M1L1 — Structure Unstructured Restaurant Data with an LLM
=========================================================
Reads the California Culinary Map (one paragraph per restaurant), converts
each paragraph into a validated JSON record via a watsonx.ai Granite model,
auto-repairs invalid model output, assigns sequential itemIds, and saves
structured_restaurant_data.json.

Reconstructed to the lab's screenshot and grading requirements
(M1L1_structure_for_loop: the loop that generates and validates JSON for
each restaurant). Shares the Lesson 1 LLM pipeline used by the M1L3 CLI.

Credentials via environment variables only (see restaurant_data_management.py).
"""

import json
from pathlib import Path

from restaurant_data_management import (
    JSON_auto_repair_prompts,
    RestaurantRecord,
    _strip_markdown_fences,
    llm_model,
    restaurant_data_structure_prompt_generation,
)
from pydantic import ValidationError

CULINARY_MAP_PATH = Path("California-Culinary-Map.txt")
OUTPUT_PATH = Path("structured_restaurant_data.json")

SYSTEM_MSG = (
    "You are a data-structuring assistant that converts restaurant "
    "descriptions into strict JSON. Return only valid JSON."
)


def load_restaurant_paragraphs(path: Path = CULINARY_MAP_PATH) -> list[str]:
    """Split the raw culinary map into one paragraph per restaurant."""
    raw_text = path.read_text(encoding="utf-8")
    return [p.strip() for p in raw_text.split("\n\n") if p.strip()]


def structure_all_restaurants(
    paragraphs: list[str], max_repair_attempts: int = 3
) -> list[str]:
    """The M1L1 core loop: for each restaurant paragraph, generate JSON with
    the LLM, validate it, and auto-repair invalid responses."""
    structured_restaurant_lists = []

    for i, paragraph in enumerate(paragraphs):
        prompt = restaurant_data_structure_prompt_generation(paragraph)
        response = llm_model(SYSTEM_MSG, prompt)

        for attempt in range(max_repair_attempts + 1):
            try:
                candidate = json.loads(_strip_markdown_fences(response))
                RestaurantRecord(**candidate)  # schema validation
                response = json.dumps(candidate)
                break
            except (json.JSONDecodeError, ValidationError) as error:
                if attempt == max_repair_attempts:
                    raise ValueError(
                        f"Restaurant {i}: could not repair LLM output "
                        f"after {max_repair_attempts} attempts: {error}"
                    )
                print(f"Restaurant {i}: invalid JSON (attempt {attempt + 1}), repairing...")
                repair_sys, repair_prompt = JSON_auto_repair_prompts(response, str(error))
                response = llm_model(repair_sys, repair_prompt)

        structured_restaurant_lists.append(response)
        print(f"Restaurant {i}: structured OK")

    return structured_restaurant_lists


def save_structured_data(
    structured_restaurant_lists: list[str], filename: Path = OUTPUT_PATH
) -> list[dict]:
    """Assign itemIds and save — verbatim per the lab's final save cell."""
    structured_restaurant_lists_json = [
        json.loads(response) for response in structured_restaurant_lists
    ]
    for i, response in enumerate(structured_restaurant_lists_json):
        response["itemId"] = 1000001 + i
        structured_restaurant_lists_json[i] = response
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(structured_restaurant_lists_json, f, indent=4)
    return structured_restaurant_lists_json


if __name__ == "__main__":
    paragraphs = load_restaurant_paragraphs()
    structured = structure_all_restaurants(paragraphs)
    records = save_structured_data(structured)
    print(f"Saved {len(records)} structured restaurant records to {OUTPUT_PATH}")
