# CityLens App

AI-powered urban scene analysis application.

## Structure

```
citylens-app/
├── backend/           # FastAPI server
│   ├── main.py        # API entry point
│   ├── src/           # Core analysis algorithms
│   └── requirements.txt
└── frontend/          # Next.js web app
    └── src/app/       # App Router pages
```

## Setup

### Backend

1. Create virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

4. Run server:
```bash
python main.py
# Or: uvicorn main:app --reload --port 8000
```

Backend runs at: http://localhost:8000

### Frontend

1. Install dependencies:
```bash
cd frontend
pnpm install
```

2. Run development server:
```bash
pnpm dev
```

Frontend runs at: http://localhost:3000

## Usage

1. Start both backend and frontend servers
2. Open http://localhost:3000 in your browser
3. Upload an urban photo (drag & drop or click to browse)
4. Click "Analyze Scene" to get AI insights
