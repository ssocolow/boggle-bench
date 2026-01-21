"""
Boggle solver script

Usage:
    python solver.py game.json dictionary.csv
"""

import json
import sys
from pathlib import Path


class TrieNode:
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True


def load_dictionary(dictionary_path: str, min_length: int = 3) -> Trie:
    """Load dictionary into a trie, skipping words shorter than min_length."""
    trie = Trie()
    with open(dictionary_path, "r") as f:
        for line in f:
            word = line.strip().upper()
            if len(word) >= min_length:
                trie.insert(word)
    return trie


def find_all_words(grid: list[list[str]], trie: Trie) -> set[str]:
    """Find all valid words on the board using DFS traversal."""
    rows = len(grid)
    cols = len(grid[0])
    found_words = set()

    # 8 directions: up, down, left, right, and 4 diagonals
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    def dfs(row: int, col: int, node: TrieNode, path: str, visited: set[tuple[int, int]]) -> None:
        """DFS from current cell, following trie paths."""
        # Get the letter(s) at this cell - handle "Qu" specially
        cell = grid[row][col]
        letters = cell.upper()  # "Qu" becomes "QU"

        # Follow the trie for each letter in this cell
        current_node = node
        for letter in letters:
            if letter not in current_node.children:
                return  # No valid words down this path
            current_node = current_node.children[letter]

        # Build the new path
        new_path = path + letters

        # If this is a complete word, add it
        if current_node.is_word:
            found_words.add(new_path)

        # If no children, no point continuing
        if not current_node.children:
            return

        # Mark current cell as visited
        visited.add((row, col))

        # Explore all neighbors
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            # Check bounds and not already visited
            if (0 <= new_row < rows and 0 <= new_col < cols
                    and (new_row, new_col) not in visited):
                dfs(new_row, new_col, current_node, new_path, visited)

        # Unmark current cell (backtrack)
        visited.remove((row, col))

    # Start DFS from each cell
    for row in range(rows):
        for col in range(cols):
            dfs(row, col, trie.root, "", set())

    return found_words


def getPointsForWord(word: str) -> int:
    """Calculate Boggle score for a word."""
    return len(word) - 2


def calculate_total_score(words: set[str]) -> int:
    """Calculate total score for all words found."""
    return sum(getPointsForWord(word) for word in words)


def main():
    if len(sys.argv) != 3:
        print("Usage: python solver.py game.json dictionary.csv")
        sys.exit(1)

    game_path = sys.argv[1]
    dictionary_path = sys.argv[2]

    # Load game data
    with open(game_path, "r") as f:
        game_data = json.load(f)

    grid = game_data["correctGrid"]

    # Load dictionary into trie
    print(f"Loading dictionary from {dictionary_path}...")
    trie = load_dictionary(dictionary_path)

    # Find all words
    print("Searching for words...")
    found_words = find_all_words(grid, trie)

    # Calculate score
    total_score = calculate_total_score(found_words)

    # Sort words for consistent output
    sorted_words = sorted(found_words)

    print(f"Found {len(sorted_words)} words with total score {total_score}")

    # Create output data
    output_data = {
        "validWords": sorted_words,
        "maxScore": total_score
    }

    # Write to gamesolved.json in same directory as input
    output_path = Path(game_path).parent / "gamesolved.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()