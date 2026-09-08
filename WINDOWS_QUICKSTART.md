# Windows Quick Start

Open the project folder in VS Code.

## 1. Backend

Open Terminal > New Terminal and run:

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```

Leave this terminal running.

## 2. Frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

## 3. API key

Create `backend/.env`:

```env
OPENAI_API_KEY=YOUR_KEY_HERE
OPENAI_MODEL=gpt-5-mini
EMBEDDING_MODEL=text-embedding-3-small
```

Never commit or share `.env`.

## 4. Test

First click "Load demo data". Then ask:
- What are customers complaining about most?
- Why are customers leaving?
- What feature has been requested most?
- Show me feedback from high-value customers.
- What should we fix first?

For model evaluation:

```powershell
cd backend
$env:RUN_AI_EVAL="1"
python tests/evaluate_1000.py
```
