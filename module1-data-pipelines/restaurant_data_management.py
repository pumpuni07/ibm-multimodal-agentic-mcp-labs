"""
Restaurant Data Management — Command-Line UI
=============================================
IBM Skills Network Lab (Module 1, Lesson 3):
"Build a Command-Line Data Management UI for Restaurant Data"

Integrates the Lesson 1 LLM-structuring pipeline (watsonx.ai Granite) with a
CLI for browsing, viewing, adding, editing, and deleting restaurant records
stored in structured_restaurant_data.json.

Credentials are read from environment variables (never hard-coded):
    WATSONX_APIKEY      - IBM Cloud API key (not needed inside the Skills
                          Network lab environment, which injects credentials)
    WATSONX_URL         - defaults to https://us-south.ml.cloud.ibm.com
    WATSONX_PROJECT_ID  - defaults to "skills-network" (the lab default)

Run the unit tests with:
    python restaurant_data_management.py
"""

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch

FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'
EXAMPLE_RESTAURANT_PARAGRAPH = (
    'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, '
    '**casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** '
    'rating, it captures the salt-air energy of the coast through its signature '
    'beer-battered snapper tacos and zesty octopus ceviche, making it a premier '
    'spot for open-air dining near the pier. Price range: $$'
)

WATSONX_URL = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_APIKEY = os.environ.get("WATSONX_APIKEY")           # optional in lab env
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "skills-network")
MODEL_ID = os.environ.get("WATSONX_MODEL_ID", "ibm/granite-3-3-8b-instruct")


# =====================================================================
# Pydantic schema — validates the JSON structure produced by the LLM
# (fields match the structured restaurant data used throughout the course)
# =====================================================================
class RestaurantRecord(BaseModel):
    name: str = Field(..., description="Restaurant name")
    neighborhood: Optional[str] = Field(None, description="Neighborhood / city")
    cuisine: Optional[str] = Field(None, description="Cuisine style")
    type: Optional[str] = Field(None, description="Venue type, e.g. taqueria")
    rating: Optional[float] = Field(None, description="Rating out of 5")
    price_range: Optional[str] = Field(None, description="e.g. $, $$, $$$")
    signature_dish: Optional[str] = Field(None, description="Signature dish(es)")
    vibes: Optional[List[str]] = Field(default_factory=list, description="Vibe tags")
    description: Optional[str] = Field(None, description="One-sentence summary")


# =====================================================================
# Helper functions (file I/O and record display)
# NOTE: the lab ships pre-defined versions of these; the implementations
# below are functionally equivalent replacements.
# =====================================================================
def load_data(file_path):
    """Load the restaurant records list from the JSON database file."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(data, file_path, backup_path):
    """Back up the current database file, then save the updated records."""
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_path)   # safety protocol: backup first
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def show_restaurant_card(res, index):
    """Pretty-print a single restaurant record."""
    print(f"\n===== Record #{index} =====")
    for key, value in res.items():
        print(f"  {key:<16}: {value}")
    print("=" * 26)


def _parse_index(raw, data):
    """Convert user input to a valid 0-based record index, or return None."""
    try:
        idx = int(str(raw).strip())
    except (ValueError, TypeError):
        return None
    if 0 <= idx < len(data):
        return idx
    return None


# =====================================================================
# Exercise 1 — LLM pipeline (integrated from Lesson 1)
# =====================================================================
def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    """Build the prompt instructing the LLM to convert an unstructured
    restaurant paragraph into a strict JSON object."""
    prompt = f"""Convert the following restaurant description into a single JSON object.

Use exactly these keys:
- "name" (string)
- "neighborhood" (string)
- "cuisine" (string)
- "type" (string, e.g. "casual taqueria", "fine dining")
- "rating" (number out of 5, e.g. 4.2; use null if not mentioned)
- "price_range" (string of dollar signs, e.g. "$$"; use null if not mentioned)
- "signature_dish" (string)
- "vibes" (array of short lowercase strings, e.g. ["sun-drenched", "casual"])
- "description" (string, one-sentence summary)

Rules:
1. Return ONLY the JSON object. No markdown fences, no commentary.
2. Use null for any value not present in the text.
3. Do not invent facts that are not in the paragraph.

Restaurant description:
{restaurant_paragraph}
"""
    return prompt


def llm_model(system_msg, prompt_txt, params=None):
    """Send a chat request to a watsonx.ai Granite model and return the
    generated text. Granite is used because it is fast and inexpensive
    for structured-extraction tasks."""
    if params is None:
        params = {"temperature": 0, "max_tokens": 1024}

    credentials = (
        Credentials(url=WATSONX_URL, api_key=WATSONX_APIKEY)
        if WATSONX_APIKEY
        else Credentials(url=WATSONX_URL)
    )
    model = ModelInference(
        model_id=MODEL_ID,
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID,
        params=params,
    )
    response = model.chat(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_txt},
        ]
    )
    return response["choices"][0]["message"]["content"]


def JSON_auto_repair_prompts(response, error_message):
    """Build system/user prompts asking the LLM to repair invalid JSON."""
    system_msg = (
        "You are a strict JSON repair assistant. You receive broken or "
        "schema-invalid JSON plus the parser/validation error, and you return "
        "ONLY the corrected JSON object — no explanations, no markdown fences."
    )
    prompt = f"""The following JSON output is invalid.

--- INVALID JSON ---
{response}

--- ERROR MESSAGE ---
{error_message}

Fix the JSON so that it parses correctly and satisfies the error above.
Return ONLY the corrected JSON object.
"""
    return system_msg, prompt


def _strip_markdown_fences(text):
    """Remove ```json ... ``` fences if the model wrapped its output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]          # drop ```json line
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    return cleaned.strip()


def new_data_entry_process(paragraph, itemId, max_repair_attempts=3):
    """Structure a new restaurant paragraph with the LLM, validate the JSON
    (Pydantic), and auto-repair invalid responses in a loop.

    Returns a validated dict with the given itemId attached.
    """
    system_msg = (
        "You are a data-structuring assistant that converts restaurant "
        "descriptions into strict JSON. Return only valid JSON."
    )
    prompt = restaurant_data_structure_prompt_generation(paragraph)
    response = llm_model(system_msg, prompt)

    for attempt in range(max_repair_attempts + 1):
        try:
            candidate = json.loads(_strip_markdown_fences(response))
            validated = RestaurantRecord(**candidate)
            record = validated.model_dump()
            record["itemId"] = itemId
            return record
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_repair_attempts:
                raise ValueError(
                    f"LLM output could not be repaired after "
                    f"{max_repair_attempts} attempts: {e}"
                )
            print(f"⚠️  Invalid JSON (attempt {attempt + 1}), auto-repairing...")
            repair_sys, repair_prompt = JSON_auto_repair_prompts(response, str(e))
            response = llm_model(repair_sys, repair_prompt)


# =====================================================================
# Exercise 2 — The main UI function
# =====================================================================
def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        if choice == '1':
            print("\n--- Current Listings ---")
            # Iterate through the records and show their names ('N/A' if absent)
            for i, res in enumerate(data):
                print(f"{i}. {res.get('name', 'N/A')}")

        elif choice == '2':
            # Get the record index, validate it, and show the card
            raw = input("Enter record index to view: ")
            idx = _parse_index(raw, data)
            if idx is not None:
                show_restaurant_card(data[idx], idx)
            else:
                print("invalid index.")

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3':  # ADD NEW DATA
                itemId = 1000000 + len(data) + 1  # item id for the new record

                # First: ask the user for a new restaurant description.
                paragraph = input("Enter the new restaurant description:\n")
                # Second: process the paragraph with the LLM pipeline.
                new_record = new_data_entry_process(paragraph, itemId)
                # Third: append the new record to the data list.
                data.append(new_record)
                # Finally: save with save_data().
                save_data(data, file_path, backup_path)

                print("✅ Restaurant added.")

            elif choice == '4':  # EDIT DATA
                raw = input("Enter record index to edit: ")
                idx = _parse_index(raw, data)
                if idx is not None:
                    record = data[idx]
                    print("Press Enter to keep the current value.")
                    for key in list(record.keys()):
                        new_value = input(f"{key} [{record[key]}]: ")
                        if new_value.strip():
                            record[key] = new_value.strip()
                    data[idx] = record
                    save_data(data, file_path, backup_path)
                    print("✅ Record updated.")
                else:
                    print("invalid index.")

            elif choice == '5':  # DELETE DATA
                raw = input("Enter record index to delete: ")
                idx = _parse_index(raw, data)
                if idx is not None:
                    removed = data.pop(idx)
                    save_data(data, file_path, backup_path)
                    print(f"✅ Record deleted: {removed.get('name', 'N/A')}")
                else:
                    print("invalid index.")

        elif choice == '6':  # EXIT
            break
        else:
            print("Invalid input.")


# =====================================================================
# Exercise 3 — Unit tests (as provided by the lab, indentation corrected)
# =====================================================================
class TestRestaurantDatabase(unittest.TestCase):

    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant.
        Inputs: '3' (Add), 'yes' (Confirm), 'New Burger Joint', '6' (Exit)
        """
        # We mock the sequence of user inputs
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal foraging and local heritage, headlined by signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience is designed to be intimate and earthy, making it a premier spot for those seeking high-quality, smokehouse-influenced cuisine in a refined, atmospheric setting.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']

        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass  # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', 1, '6']

        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass  # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        Inputs: '5' (Delete), 'no' (Cancel), '6' (Exit)
        """
        mock_input.side_effect = ['5', 'no', '6']

        manage_restaurants(self.test_file, self.test_file_backup)

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(len(data), 1)  # Data should remain unchanged
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())


if __name__ == "__main__":
    unittest.main()  # Unit Test
    # manage_restaurants(FILEPATH, BACKUP_PATH)  # Actual UI Call
