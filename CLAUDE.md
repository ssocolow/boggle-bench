# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Boggle-eval benchmarks LLMs on 5x5 Boggle word games, evaluating two tasks: (1) transcribing Boggle boards from images, and (2) finding valid words. Results are displayed on a static web dashboard.

Live demo: https://simonsocolow.com/boggle-eval

## Commands

### Running Evaluations

```bash
# List available models
python eval.py list-models

# Transcribe a board image
python eval.py transcribe --image boggle1.png --models gpt-4o,claude-3.5-sonnet

# Find words on a grid (use semicolons to separate rows)
python eval.py find-words --grid "R,N,E,A,N;O,M,E,N,C;Y,T,E,W,E;L,I,P,I,E;E,Qu,F,T,O" --models gpt-4o

# Full evaluation (transcription + word-finding)
python eval.py full-eval \
  --image boggle1.png \
  --correct-grid "R,N,E,A,N;O,M,E,N,C;Y,T,E,W,E;L,I,P,I,E;E,Qu,F,T,O" \
  --models gpt-4o,claude-3.5-sonnet \
  --output-dir data/game1/models
```

### Serving the Frontend

```bash
python -m http.server 8000
```

## Architecture

**Backend (`eval.py`)**: Python script that calls LLMs via OpenRouter API. Sends board images for transcription and grids for word-finding. Outputs result JSON files to `data/game1/models/`.

**Frontend (`index.html`, `script.js`, `style.css`)**: Vanilla JS single-page app that loads game data and model results from JSON files, renders a comparison dashboard with score charts and model cards.

**Data flow**:
1. `eval.py` calls OpenRouter API → generates `data/game1/models/{model}.json`
2. Frontend fetches `data/game1/game.json` (correct answers) and `models/index.json` (file listing)
3. Loads all model results in parallel, renders score chart and comparison cards

## Key Files

- `eval.py` - CLI for running evaluations (requires `OPENROUTER_API_KEY` env var)
- `data/game1/game.json` - Correct grid, max score, valid words for a game
- `data/game1/models/index.json` - Array of model result filenames
- `data/game1/models/*.json` - Individual model results (transcription, words found, score)

## Dependencies

- Python: `httpx` (for API calls)
- External: OpenRouter API key in `OPENROUTER_API_KEY` environment variable
- Optional: `dictionary.csv` (Scrabble dictionary, download from https://github.com/zeisler/scrabble/blob/master/db/dictionary.csv)

## Notes

- Grid format uses "Qu" for the Q tile (not just "Q")
- Frontend is hardcoded to `game1` - modify `script.js` to support multiple games
- There's a known bug in word scoring calculation (see TODO in eval.py line 262)
