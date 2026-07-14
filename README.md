# IBM Multimodal, Agentic & MCP Labs

Hands-on lab implementations from the IBM course covering LLM-based data
structuring, multimodal RAG, multi-agent recommendation systems, and
Model Context Protocol (MCP) integration — built around a California
restaurant recommendation application ("Connoisseur").

## Lab status

| Module | Lab | File | Status |
|--------|-----|------|--------|
| M1L1 | Structure Unstructured Restaurant Data with an LLM | `module1-data-pipelines/structure_restaurant_data.py` | ✅ Tests passing (M1 suite 11/11) |
| M1L2 | Process Multimodal Data with LLMs (vision captioning) | `module1-data-pipelines/process_multimodal_reviews.py` | ✅ Tests passing (M1 suite 11/11) |
| M1L3 | Command-Line Data Management UI for Restaurant Data | `module1-data-pipelines/restaurant_data_management.py` | ✅ Unit tests passing (2/2) |
| M4L1 | Build an MCP Server | `module4-mcp/server.py` | ✅ Verified via `test.py` |
| M4L2 | Build an MCP Client | `module4-mcp/client.py` | ✅ All 3 demos + discovery verified |
| M4L3 | Build a Full MCP Application (Gradio host) | `module4-mcp/app.py` | ✅ UI builds; ReAct loop wired |
| M3L1 | Design Specialized Agents for a Recommendation System | `module3-multi-agent/design_agents.py` | ✅ Tests passing (8/8) |
| M2L1–M2L3, M3L2–M3L3 | Multimodal vector indexing, retrieval, fusion; multi-agent workflow & chatbot | — | 🔜 To be added |

---

## Module 1 — LLM Data Pipelines

### M1L3: Restaurant Data Management CLI

A Python CLI to browse, view, add, edit, and delete restaurant records in a
structured JSON database. New records are entered as free-text paragraphs and
structured automatically by an LLM pipeline:

- **Prompt generation** — instructs a watsonx.ai Granite model to convert an
  unstructured restaurant description into a strict JSON schema
- **Pydantic validation** — every LLM response is validated against a
  `RestaurantRecord` schema before it touches the database
- **JSON auto-repair loop** — malformed model output is sent back to the LLM
  with the parser/validation error for correction (up to 3 attempts)
- **Safety protocols** — write operations require explicit confirmation, and
  the database file is backed up before every save

```bash
cd module1-data-pipelines
python3.11 -m venv venv && source venv/bin/activate
pip install ibm-watsonx-ai==1.4.7 pydantic==2.12.4
python restaurant_data_management.py   # runs the unit tests (expected: OK, 2 tests)
```

To use the interactive UI instead, swap the two lines at the bottom of the
file (comment `unittest.main()`, uncomment `manage_restaurants(...)`).

---

## Module 4 — Model Context Protocol (MCP)

An end-to-end MCP stack: a FastMCP **server** exposing restaurant data as a
resource and three tools, a protocol-complete **client** implementing roots
and sampling callbacks, and a Gradio **host** application driving a ReAct
agent loop with a watsonx.ai LLM.

```
module4-mcp/
├── server.py          # M4L1 — FastMCP server: 1 resource, 3 tools
├── test.py            # M4L1 — stdio smoke test (screenshot source)
├── client.py          # M4L2 — MCP client: roots, sampling, discovery, demos
├── app.py             # M4L3 — Gradio + watsonx ReAct agent host
└── download_data.sh   # fetches the official course data files
```

### Setup

```bash
cd module4-mcp
pip install virtualenv && virtualenv .venv && source .venv/bin/activate
pip install fastmcp==3.1.0 mcp==1.25.0 anthropic==0.84.0 \
            gradio==6.9.0 langchain-ibm==1.0.4 langchain-core==1.2.18
bash download_data.sh   # California-Culinary-Map.txt + 2 JSON datasets
```

### M4L1 — Server

Exposes `culinary-map://california` as a resource plus three tools:
`get_restaurant_info` (structured name search), `recommend_by_vibe`
(two-pass search over vibe tags and raw text), and `get_review` (full
review records). Verify with:

```bash
python test.py   # prints a JSON "found" result for the query "Iron"
```

### M4L2 — Client

Launches the server as a subprocess over stdio and implements the two
optional client-side MCP callbacks: **roots** (restricting server file
access to the project directory) and **sampling** (fulfilling
server-delegated LLM requests via the Anthropic API). Runs three tool
demos, then discovers and asserts all tools, resources, and roots.

```bash
export ANTHROPIC_API_KEY="your-key"   # used only for sampling requests
python client.py
```

### M4L3 — Host application

A Gradio chat UI ("Connoisseur Companion") backed by IBM Granite via
`ChatWatsonx`. On every turn the host discovers the server's tools at
runtime, converts MCP schemas to OpenAI-style tool definitions, and runs
a ReAct loop — reason → tool call → observe → repeat — until the model
returns a final natural-language answer.

```bash
export WATSONX_AI_PROJECT_ID="your-project-id"
gradio app.py   # opens local + public Gradio URLs
```

---

## Credentials

All credentials are read from environment variables — nothing is
hard-coded: `WATSONX_APIKEY`, `WATSONX_URL`, `WATSONX_PROJECT_ID` /
`WATSONX_AI_PROJECT_ID`, `ANTHROPIC_API_KEY`. Inside the IBM Skills
Network lab environment, watsonx credentials are injected automatically.

## License

Lab content is licensed under Apache 2.0 (per the original IBM course
materials). Implementations are my own.

## Author

**Jack Pumpuni Frimpong-Manso**
[GitHub](https://github.com/pumpuni07) · [LinkedIn](https://www.linkedin.com/in/jack-pumpuni-frimpong-manso) · [Portfolio](https://jackpumpunifrimpongmanso.base44.app)
