#!/usr/bin/env python3
"""
Boggle Eval Script

Uses OpenRouter to evaluate multiple AI models on:
1. Transcribing a Boggle board from an image
2. Finding valid words given a correct transcription

Usage:
    python eval.py transcribe --image boggle1.png --models gpt-4o,claude-3.5-sonnet
    python eval.py find-words --grid "R,E,Z,A,E;O,N,E,M,Z;Y,T,E,W,M;Qu,I,I,P,I;W,O,F,L,O" --models gpt-4o,claude-3.5-sonnet

Environment:
    OPENROUTER_API_KEY - Your OpenRouter API key
"""

import argparse
import base64
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import httpx

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model mapping: short name -> OpenRouter model ID
MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gemini-1.5-pro": "google/gemini-pro-1.5",
    "deepseek-v3": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-r1",
    "llama-3.2-90b": "meta-llama/llama-3.2-90b-vision-instruct",
    "qwen-vl-plus": "qwen/qwen-vl-plus",
}

TRANSCRIPTION_PROMPT = """Look at this Boggle game board image. Transcribe the 5x5 grid of letters exactly as shown.

Rules:
- The grid is 5 rows by 5 columns
- Each cell contains a single letter, except "Qu" which appears together on one die
- Transcribe exactly what you see, preserving the case

Return your answer as a JSON array of arrays, like this:
[
  ["A", "B", "C", "D", "E"],
  ["F", "G", "H", "I", "J"],
  ["K", "L", "M", "N", "O"],
  ["P", "Qu", "R", "S", "T"],
  ["U", "V", "W", "X", "Y"]
]

Return ONLY the JSON array, no other text."""

FIND_WORDS_PROMPT = """You are playing Boggle. Here is the 5x5 letter grid:

{grid_display}

Boggle Rules:
1. Form words by connecting adjacent letters (horizontally, vertically, or diagonally)
2. Each die (letter) can only be used once per word
3. Words must be at least 3 letters long
4. Words must be valid English dictionary words (common nouns, verbs, adjectives, adverbs)
5. No proper nouns, abbreviations, or contractions
6. "Qu" counts as two letters but occupies one cell

Find as many valid words as you can. Focus on quality over quantity - only include words you're confident are:
1. Valid English words
2. Actually traceable on the board following adjacency rules

Return your answer as a JSON object with this format:
{{
  "words": ["WORD1", "WORD2", "WORD3", ...]
}}

Return ONLY the JSON object, no other text."""


def get_api_key():
    """Get OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("Error: OPENROUTER_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    return key


def encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64 and determine media type."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/png")

    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, media_type


def call_openrouter(model_id: str, messages: list, api_key: str) -> str:
    """Call OpenRouter API and return the response text."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/simonw/boggle-eval",
    }

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 4096,
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    return data["choices"][0]["message"]["content"]


def parse_json_response(text: str) -> dict | list | None:
    """Extract JSON from response text, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block_match:
        text = code_block_match.group(1).strip()

    # Try to parse as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array or object in the text
        json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    return None


def transcribe_board(image_path: str, model_name: str, api_key: str) -> dict:
    """Transcribe a Boggle board image using a specific model."""
    model_id = MODELS.get(model_name)
    if not model_id:
        return {"error": f"Unknown model: {model_name}"}

    image_data, media_type = encode_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}"
                    }
                },
                {
                    "type": "text",
                    "text": TRANSCRIPTION_PROMPT
                }
            ]
        }
    ]

    print(f"  Calling {model_name}...", file=sys.stderr)

    try:
        response_text = call_openrouter(model_id, messages, api_key)
        grid = parse_json_response(response_text)

        if grid is None:
            return {"error": "Failed to parse response", "raw_response": response_text}

        if not isinstance(grid, list) or len(grid) != 5:
            return {"error": "Invalid grid format", "raw_response": response_text}

        return {"grid": grid, "raw_response": response_text}

    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP error: {e.response.status_code}", "details": str(e)}
    except Exception as e:
        return {"error": str(e)}


def find_words(grid: list[list[str]], model_name: str, api_key: str) -> dict:
    """Find valid Boggle words using a specific model."""
    model_id = MODELS.get(model_name)
    if not model_id:
        return {"error": f"Unknown model: {model_name}"}

    # Format grid for display
    grid_display = "\n".join(
        "  ".join(cell.ljust(2) for cell in row)
        for row in grid
    )

    prompt = FIND_WORDS_PROMPT.format(grid_display=grid_display)

    messages = [
        {"role": "user", "content": prompt}
    ]

    print(f"  Calling {model_name}...", file=sys.stderr)

    try:
        response_text = call_openrouter(model_id, messages, api_key)
        result = parse_json_response(response_text)

        if result is None:
            return {"error": "Failed to parse response", "raw_response": response_text}

        words = result.get("words", []) if isinstance(result, dict) else result

        if not isinstance(words, list):
            return {"error": "Invalid response format", "raw_response": response_text}

        # Normalize words to uppercase
        words = [w.upper() for w in words if isinstance(w, str)]

        return {"words": words, "raw_response": response_text}

    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP error: {e.response.status_code}", "details": str(e)}
    except Exception as e:
        return {"error": str(e)}


def parse_grid_string(grid_str: str) -> list[list[str]]:
    """Parse a grid string like 'A,B,C,D,E;F,G,H,I,J;...' into a 2D array."""
    rows = grid_str.split(";")
    return [row.split(",") for row in rows]


def calculate_word_score(words: list[str]) -> int:
    """Calculate Boggle score for a list of words."""
    score = 0
    for word in words:
        length = len(word.replace("QU", "Q"))  # Qu counts as 2 letters but 1 die
        if length <= 2:
            continue
        elif length == 3 or length == 4:
            score += 1 # TODO this is wrong. if length is 3 score should be 1. if length is four score should be 2
        elif length == 5:
            score += 2
        elif length == 6:
            score += 3
        elif length == 7:
            score += 5
        else:
            score += 11
    return score


def generate_model_json(model_name: str, transcription_grid: list, words: list, word_score: int) -> dict:
    """Generate a model JSON file for the site."""
    # Map short names to display names
    display_names = {
        "gpt-4o": "GPT-4o",
        "gpt-4o-mini": "GPT-4o Mini",
        "claude-3.5-sonnet": "Claude 3.5 Sonnet",
        "claude-3-opus": "Claude 3 Opus",
        "claude-3-haiku": "Claude 3 Haiku",
        "gemini-2.0-flash": "Gemini 2.0 Flash",
        "gemini-1.5-pro": "Gemini 1.5 Pro",
        "deepseek-v3": "DeepSeek-V3",
        "deepseek-r1": "DeepSeek-R1",
        "llama-3.2-90b": "Llama 3.2 90B",
        "qwen-vl-plus": "Qwen VL Plus",
    }

    return {
        "model": display_names.get(model_name, model_name),
        "date": date.today().isoformat(),
        "transcriptionGrid": transcription_grid,
        "wordsFound": sorted(words),
        "wordScore": word_score,
    }


def cmd_transcribe(args):
    """Run transcription evaluation."""
    api_key = get_api_key()
    models = [m.strip() for m in args.models.split(",")]

    print(f"Transcribing {args.image} with {len(models)} models...", file=sys.stderr)

    results = {}
    for model in models:
        result = transcribe_board(args.image, model, api_key)
        results[model] = result

        if "grid" in result:
            print(f"  {model}: Success", file=sys.stderr)
        else:
            print(f"  {model}: Error - {result.get('error', 'Unknown')}", file=sys.stderr)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(results, indent=2))


def cmd_find_words(args):
    """Run word-finding evaluation."""
    api_key = get_api_key()
    models = [m.strip() for m in args.models.split(",")]
    grid = parse_grid_string(args.grid)

    print(f"Finding words with {len(models)} models...", file=sys.stderr)

    results = {}
    for model in models:
        result = find_words(grid, model, api_key)
        results[model] = result

        if "words" in result:
            print(f"  {model}: Found {len(result['words'])} words", file=sys.stderr)
        else:
            print(f"  {model}: Error - {result.get('error', 'Unknown')}", file=sys.stderr)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(results, indent=2))


def cmd_full_eval(args):
    """Run full evaluation: transcribe then find words."""
    api_key = get_api_key()
    models = [m.strip() for m in args.models.split(",")]

    # Load correct grid for word finding
    correct_grid = parse_grid_string(args.correct_grid)

    print(f"Running full evaluation with {len(models)} models...", file=sys.stderr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = []

    for model in models:
        print(f"\nEvaluating {model}...", file=sys.stderr)

        # Step 1: Transcribe
        print("  Step 1: Transcribing board...", file=sys.stderr)
        trans_result = transcribe_board(args.image, model, api_key)

        if "error" in trans_result:
            print(f"  Transcription failed: {trans_result['error']}", file=sys.stderr)
            continue

        transcription_grid = trans_result["grid"]

        # Step 2: Find words (using correct grid)
        print("  Step 2: Finding words...", file=sys.stderr)
        words_result = find_words(correct_grid, model, api_key)

        if "error" in words_result:
            print(f"  Word finding failed: {words_result['error']}", file=sys.stderr)
            continue

        words = words_result["words"]
        word_score = calculate_word_score(words)

        # Generate model JSON
        model_data = generate_model_json(model, transcription_grid, words, word_score)

        # Save to file
        filename = f"{model.replace('.', '-').replace('/', '-')}.json"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

        index.append(filename)
        print(f"  Saved to {filepath}", file=sys.stderr)
        print(f"  Transcription errors: {25 - sum(1 for i in range(5) for j in range(5) if transcription_grid[i][j] == correct_grid[i][j])}", file=sys.stderr)
        print(f"  Words found: {len(words)}, Score: {word_score}", file=sys.stderr)

    # Save index
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nIndex saved to {index_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Boggle evaluation using OpenRouter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Transcribe command
    trans_parser = subparsers.add_parser("transcribe", help="Transcribe a Boggle board from an image")
    trans_parser.add_argument("--image", required=True, help="Path to the Boggle board image")
    trans_parser.add_argument("--models", required=True, help="Comma-separated list of models to use")
    trans_parser.add_argument("--output", help="Output file (default: stdout)")
    trans_parser.set_defaults(func=cmd_transcribe)

    # Find words command
    words_parser = subparsers.add_parser("find-words", help="Find valid words on a Boggle board")
    words_parser.add_argument("--grid", required=True, help="Grid as semicolon-separated rows, comma-separated cells")
    words_parser.add_argument("--models", required=True, help="Comma-separated list of models to use")
    words_parser.add_argument("--output", help="Output file (default: stdout)")
    words_parser.set_defaults(func=cmd_find_words)

    # Full evaluation command
    full_parser = subparsers.add_parser("full-eval", help="Run full evaluation (transcribe + find words)")
    full_parser.add_argument("--image", required=True, help="Path to the Boggle board image")
    full_parser.add_argument("--correct-grid", required=True, help="Correct grid for word finding")
    full_parser.add_argument("--models", required=True, help="Comma-separated list of models to use")
    full_parser.add_argument("--output-dir", required=True, help="Output directory for model JSON files")
    full_parser.set_defaults(func=cmd_full_eval)

    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available models")
    list_parser.set_defaults(func=lambda args: print("\n".join(f"  {k}: {v}" for k, v in MODELS.items())))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
