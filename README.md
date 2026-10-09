# chessml

A from-scratch chess rules engine in pure Python (plus numpy), built to be a training
environment for ML agents. Rules are complete: castling, en passant, promotion, checkmate,
stalemate, fifty-move rule, threefold repetition, and insufficient material. Move
generation is verified with perft against the standard reference positions.

## Setup

```sh
pip install -e ".[dev]"
```

## Play in the terminal

```sh
python -m chessml                          # you (white) vs a random bot
python -m chessml --white random --black human
python -m chessml --white human --black human
python -m chessml --white random --black random --seed 1
python -m chessml --fen "8/P6k/8/8/8/8/8/K7 w - - 0 1"
```

Enter moves as UCI (`e2e4`, `e7e8q`) or SAN (`Nf3`, `exd5`, `O-O`, `e8=Q`).
Commands: `moves`, `fen`, `undo`, `help`, `quit`.

## Use as an ML environment

```python
import numpy as np
from chessml import ChessEnv

env = ChessEnv(max_plies=400)
obs, info = env.reset()                 # obs: float32 (19, 8, 8)
done = False
while not done:
    legal = np.flatnonzero(info["legal_mask"])   # mask over 4672 actions
    action = np.random.choice(legal)             # your policy goes here
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
print(info["outcome"])
```

- **Perspective:** observations and actions are always from the side to move (the board is
  mirrored for black), so one network can play both colors.
- **Observation** `(19, 8, 8)`: 6 planes for our pieces, 6 for theirs, side-to-move,
  4 castling-rights planes, en passant square, halfmove clock / 100.
- **Actions** `4672 = 64 × 73` (AlphaZero layout): 56 queen-like moves, 8 knight moves and
  9 underpromotions per from-square. Use `info["legal_mask"]` or `env.legal_actions()`
  to mask illegal ones. `env.step()` raises `ValueError` on an illegal action.
- **Reward:** from the perspective of the player who just moved: `+1` for delivering
  checkmate, otherwise `0`. Draws are applied automatically (no claiming needed).
- `truncated` becomes true when `max_plies` is reached without a result.

Lower-level API: `Board` (`push`, `pop`, `legal_moves`, `fen`, `outcome`, `copy`, Zobrist
`hash`), `encode_board`, `move_to_action`, `action_to_move`, `legal_action_mask`,
`move_to_san`, `parse_san`, `perft`.

## Training (Supervised model)

```sh
python -m chessml_train.lichess --month 2026-08 --games 500000 --out data/lichess-2026-08   # build dataset
python -m chessml_train.train --data data/lichess-2026-08 --out runs/sl-10x128              # train (resumable)
python -m chessml_train.arena --model runs/sl-10x128/best.pt --games 100                    # vs RandomPlayer
```

On Colab use `notebooks/train_supervised.ipynb`; checkpoints go to Google Drive and a rerun resumes.

## Explanations (LLM)

`pip install -e ".[llm]"`, then either run [Ollama](https://ollama.com) with `ollama pull qwen3:8b`, or
use Hugging Face transformers in-process (downloads `Qwen/Qwen3-1.7B` on first use):

```sh
python -m chessml_train.explain --model runs/sl-10x128/best.pt --llm ollama --show-facts
python -m chessml_train.play --model runs/sl-10x128/best.pt --simulations 800 --device cuda --llm hf
```

The LLM only puts Search's findings into words and a guardrail rejects invented moves (docs/adr/0004).

## Tests

```sh
python -m pytest -m "not slow"    # ~20s
python -m pytest                  # includes deep perft (millions of nodes, ~1 min)
```

## Layout

| Module        | Purpose                                                    |
|---------------|------------------------------------------------------------|
| `constants`   | piece codes, square helpers, precomputed attack tables     |
| `move`        | `Move` dataclass, UCI conversion                           |
| `board`       | position state, FEN, push/pop, Zobrist hashing, attacks    |
| `movegen`     | legal move generation, perft                               |
| `rules`       | game outcome detection                                     |
| `san`         | SAN formatting and parsing                                 |
| `encoding`    | numpy observation and action-space encoding                |
| `env`         | `ChessEnv` gym-style environment                           |
| `players`     | `RandomPlayer`, `HumanPlayer`                              |
| `cli`         | terminal game                                              |

`chessml_train/` holds the training code (PyTorch) and is kept out of `chessml` so the
engine stays numpy-only. Install it with `pip install -e ".[train]"`; on Windows with an
NVIDIA GPU, first install CUDA torch: `pip install torch --index-url https://download.pytorch.org/whl/cu130`.
Vocabulary is in [CONTEXT.md](CONTEXT.md) and design decisions in [docs/adr/](docs/adr/).
