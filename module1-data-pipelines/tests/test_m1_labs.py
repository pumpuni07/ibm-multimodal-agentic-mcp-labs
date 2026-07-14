"""Tests for M1L1 (structuring loop) and M1L2 (vision captioning loop).
Real pipeline logic throughout; only live watsonx calls are mocked."""

import base64
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import structure_restaurant_data as m1l1
import process_multimodal_reviews as m1l2

VALID = json.dumps({
    "name": "Iron & Embers", "neighborhood": "DTLA", "cuisine": "New American",
    "type": "smokehouse", "rating": 4.6, "price_range": "$$$",
    "signature_dish": "Brisket", "vibes": ["moody"], "description": "Moody smokehouse.",
})
BROKEN = '```json\n{"name": "Iron & Embers",}\n```'


# ---------- M1L1: paragraph loading ----------

def test_load_paragraphs_splits_on_blank_lines(tmp_path):
    f = tmp_path / "map.txt"
    f.write_text("First restaurant.\n\nSecond restaurant.\n\n\nThird.")
    assert m1l1.load_restaurant_paragraphs(f) == [
        "First restaurant.", "Second restaurant.", "Third."
    ]


# ---------- M1L1: the structuring for-loop ----------

def test_loop_structures_every_paragraph():
    with patch.object(m1l1, "llm_model", return_value=VALID) as mock:
        out = m1l1.structure_all_restaurants(["para one", "para two", "para three"])
    assert len(out) == 3
    assert mock.call_count == 3
    assert all(json.loads(r)["name"] == "Iron & Embers" for r in out)


def test_loop_repairs_invalid_json_then_continues():
    responses = iter([BROKEN, VALID, VALID])
    with patch.object(m1l1, "llm_model", side_effect=lambda *a, **k: next(responses)) as mock:
        out = m1l1.structure_all_restaurants(["para one", "para two"])
    assert len(out) == 2
    assert mock.call_count == 3  # 1 initial + 1 repair + 1 for second paragraph


def test_loop_raises_after_exhausting_repair_attempts():
    with patch.object(m1l1, "llm_model", return_value="not json at all"):
        with pytest.raises(ValueError, match="could not repair"):
            m1l1.structure_all_restaurants(["bad"], max_repair_attempts=2)


def test_loop_rejects_schema_invalid_records():
    missing_name = json.dumps({"cuisine": "Thai"})
    responses = iter([missing_name, VALID])
    with patch.object(m1l1, "llm_model", side_effect=lambda *a, **k: next(responses)):
        out = m1l1.structure_all_restaurants(["para"])
    assert json.loads(out[0])["name"] == "Iron & Embers"


# ---------- M1L1: itemId assignment and save (lab's verbatim cell) ----------

def test_save_assigns_sequential_item_ids_from_1000001(tmp_path):
    out_file = tmp_path / "structured.json"
    records = m1l1.save_structured_data([VALID, VALID, VALID], out_file)
    assert [r["itemId"] for r in records] == [1000001, 1000002, 1000003]
    saved = json.loads(out_file.read_text())
    assert saved == records


# ---------- M1L2: image encoding ----------

def test_encode_image_round_trips(tmp_path):
    img = tmp_path / "dish.jpg"
    img.write_bytes(b"jpeg-bytes")
    assert base64.b64decode(m1l2.encode_image(str(img))) == b"jpeg-bytes"


# ---------- M1L2: the captioning for-loop ----------

def test_caption_loop_assigns_captions_to_every_review_with_image(tmp_path):
    img = tmp_path / "dish.jpg"
    img.write_bytes(b"pixels")
    reviews = [
        {"restaurant_name": "A", "image_path": str(img)},
        {"restaurant_name": "B", "image_path": str(img)},
    ]
    with patch.object(m1l2, "vision_llm_caption", return_value="A golden crusted pie.") as mock:
        out = m1l2.caption_all_recipes(reviews)
    assert mock.call_count == 2
    assert all(r["image_description"] == "A golden crusted pie." for r in out)


def test_caption_loop_marks_missing_images_na():
    reviews = [{"restaurant_name": "A", "image_path": "/nope/missing.jpg"},
               {"restaurant_name": "B"}]
    with patch.object(m1l2, "vision_llm_caption") as mock:
        out = m1l2.caption_all_recipes(reviews)
    mock.assert_not_called()
    assert all(r["image_description"] == "N/A" for r in out)


def test_save_augmented_reviews_writes_json(tmp_path):
    out_file = tmp_path / "augmented.json"
    data = [{"restaurant_name": "A", "image_description": "cap"}]
    m1l2.save_augmented_reviews(data, out_file)
    assert json.loads(out_file.read_text()) == data


def test_no_hardcoded_credentials_in_either_module():
    for module in (m1l1, m1l2):
        source = Path(module.__file__).read_text()
        assert "os.environ" in source or "restaurant_data_management" in source
        assert "sk-" not in source
