# chessml

A from-scratch chess rules engine and training environment, used to train chess-playing neural networks.

## Language

### Models

**Supervised model**:
A network trained to imitate human moves and predict game results from recorded human games.
_Avoid_: Human model, imitation AI, SL agent

**Zero model**:
A network trained only from its own self-play, starting from random weights with no human game data or hand-written chess knowledge beyond the rules.
_Avoid_: RL model, AlphaZero model, self-taught AI

**Self-play**:
Games in which a model plays both sides against itself to produce its own training data.

**Tabula rasa**:
The constraint that a Zero model never learns from human games or from the Supervised model's weights or outputs.
_Avoid_: From scratch (ambiguous with "the engine is written from scratch")

### Network and search

**Policy**:
A model's probability for each legal move in a position, as seen by the side to move.
_Avoid_: Move prediction, move scores

**Value**:
A model's estimated win/draw/loss probabilities for a position, from the side to move's point of view. Always about a position, never about a move.
_Avoid_: Evaluation, score, eval (when meaning the network output)

**Search**:
Monte Carlo tree search guided by Policy and Value, used to choose moves. Its per-move visit count and average result are the only per-move judgments of a move's outcome.
_Avoid_: Lookahead, engine (the rules engine is a different thing)

**Full search**:
A self-play move searched with the full simulation budget; only these moves become Policy training targets.

**Quick search**:
A self-play move searched with a small simulation budget just to advance the game; never a Policy training target.
_Avoid_: Fast move, cheap search

**Random opening**:
The 0–8 uniformly random legal moves that start each self-play game before Search takes over. It varies the starting positions without using human knowledge.
_Avoid_: Opening book (a book is human knowledge)

**Truncated game**:
A self-play game stopped at the ply cap without a result; it is scored as a draw.
_Avoid_: Unfinished game, timeout

### Evaluation

**Opponent ladder**:
The fixed, ordered set of opponents used to measure a model's strength: RandomPlayer, a material-greedy bot, Stockfish at limited strength levels, and the human author.
Evaluation games against the ladder are never training data.
_Avoid_: Benchmark, test suite
