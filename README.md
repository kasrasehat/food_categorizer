# food_categorizer

Clean restaurant/menu product names and classify them into predefined food groups.

This repo provides:
- A **FastAPI** service for single-text categorization and batch file processing (CSV/XLSX)
- Scripts to build a **Chroma** vector database from your food-group dataset (RAG-style retrieval + voting)
- Optional **LLM translation** (OpenAI) for non-English product names before categorization

---

### What it does

- **Normalize product names**
  - Removes units/sizes, cleans separators, and normalizes Unicode
  - Includes mojibake repair for common Windows encoding issues (Arabic CP1256/Latin-1/CP1252 cases)
- **Categorize**
  - Embeds product names with OpenAI embeddings
  - Retrieves nearest neighbors from a Chroma vector DB (built from `data/food_group.json`)
  - Votes the best category and returns a lightweight probability score
- **Process CSV/XLSX**
  - Reads uploaded `csv` or `xlsx`
  - Requires an `ACTUAL_PRODUCT_NAME` column
  - Outputs the same file type with added columns:
    - `PROCESSED_PRODUCT_NAME`
    - `PRODUCT_LABEL`
    - `LABEL_PROBABILITY`

---

### Repository layout (high level)

- `main.py`: FastAPI app entrypoint (defaults to port `8004`)
- `app/api/v1/`: API routes/endpoints
  - `GET /api/v1/health`
  - `POST /api/v1/categorize`
  - `POST /api/v1/process-file` (multipart upload)
- `utils/`: core logic
  - `categorizer.py`: retrieval + voting category selection
  - `product_name_processing.py`: normalization + language detection + mojibake fixes
  - `lfspan.py`: loads Chroma DB at startup and caches the state
  - `tabular_io.py`: CSV/XLSX read/write helpers
  - `llm_translate.py`: OpenAI LLM translation to English (batch)
- `create_KB/`: scripts to build the dataset + vector DB
  - `create_chunks.py`: build `data/food_group.json` from `data/Product_Labels.xlsx`
  - `create_vectordb.py`: build Chroma DB under `data/vectordbs/chromadb_food_group`
- `data/`: datasets and persisted DB
  - `Product_Labels.xlsx`
  - `food_group.json`
  - `vectordbs/chromadb_food_group/`

---

### Requirements

- **Python 3.12**
- An **OpenAI API key** (needed for:
  - embeddings used by categorization, and
  - optional translation used by `/process-file`)

Dependencies are listed in `requirements.txt`.

---

### Setup (Windows / PowerShell)

Create and activate a virtual environment:

```powershell
cd E:\codes_py\food_categorizer
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

### Configuration (.env)

Create a `.env` in the project root:

```dotenv
OPENAI_API_KEY=your_key_here
```

Optional environment variables:
- `APP_NAME`: FastAPI title (default: `food_categorizer`)
- `APP_VERSION`: API version string (default: `0.1.0`)
- `RAG_EMBEDDING_MODEL`: embedding model (default: `text-embedding-3-small`)
- `CHROMA_FOOD_GROUP_DB_DIR`: path to Chroma persistence directory  
  - default: `data/vectordbs/chromadb_food_group`
- `TEXT_COLLECTION_NAME`: Chroma collection name (default: `food_group`)

Notes:
- `.env` is loaded automatically via `utils/settings.py` and `create_KB/create_vectordb.py`.
- Logs go to `utils/data/log.txt`.

---

### Build the knowledge base (vector DB)

#### 1) Generate `food_group.json` from Excel

```powershell
cd E:\codes_py\food_categorizer
.\.venv\Scripts\Activate.ps1
python -m create_KB.create_chunks --input .\data\Product_Labels.xlsx --output .\data\food_group.json
```

This produces a JSON array of objects like:

```json
{ "name": "americano", "group": "beverages" }
```

#### 2) Build the Chroma vector DB

```powershell
cd E:\codes_py\food_categorizer
.\.venv\Scripts\Activate.ps1
python .\create_KB\create_vectordb.py --chunks .\data\food_group.json --db-dir .\data\vectordbs\chromadb_food_group --collection-name food_group
```

If `OPENAI_API_KEY` is not in your environment or `.env`, you can also pass:

```powershell
python .\create_KB\create_vectordb.py --openai-api-key "your_key_here"
```

---

### Run the API

```powershell
cd E:\codes_py\food_categorizer
.\.venv\Scripts\Activate.ps1
python .\main.py
```

Then open:
- Swagger UI: `http://localhost:8004/docs`

---

### API usage

#### Health check

```bash
GET /api/v1/health
```

#### Categorize a single text

```bash
POST /api/v1/categorize
Content-Type: application/json

{"text":"Americano iced"}
```

Response:

```json
{"probability":0.83,"category":"beverages"}
```

#### Process a CSV/XLSX file

Upload a file containing column `ACTUAL_PRODUCT_NAME`.

Output adds:
- `PROCESSED_PRODUCT_NAME`
- `PRODUCT_LABEL`
- `LABEL_PROBABILITY`

Example (PowerShell):

```powershell
curl -X POST "http://localhost:8004/api/v1/process-file" `
  -F "file=@.\data\your_file.xlsx"
```

---

### Troubleshooting

- **“OPENAI_API_KEY environment variable is not set”**
  - Add `OPENAI_API_KEY=...` to `.env` at the repo root, or set it as an environment variable.
- **Vector DB loads but returns “other” often**
  - Rebuild the DB from updated `food_group.json`
  - Ensure `TEXT_COLLECTION_NAME` and `CHROMA_FOOD_GROUP_DB_DIR` match what you built
- **CSV encoding issues (garbled characters)**
  - The CSV reader tries `utf-8-sig`, `utf-8`, `cp1252`, and `latin-1`
  - The name normalizer includes mojibake repair for common Windows cases

---

### License

MIT (see `LICENSE`).
