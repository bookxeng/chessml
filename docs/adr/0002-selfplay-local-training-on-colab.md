# Self-play on the local PC, training on Colab, data via Google Drive

Free Colab gives a good GPU but only about 2 CPU cores, and our rules engine is pure Python and CPU-bound, so Colab is a poor place for Search-heavy self-play. Self-play therefore runs on the author's PC (16 cores, with an RTX 3050 doing batched network evaluation), and training runs on Colab. The two exchange data through a Google Drive for desktop folder (`data/`, `selfplay/`, `checkpoints/`). Self-play writes small chunks of about 1,000 games each so syncs stay incremental, and training saves checkpoints to Drive often, because Colab sessions are routinely killed.

## Considered Options

- Everything on Colab: rejected, because self-play would get about 1/8 of the local throughput.
- Hugging Face Hub or a cloud bucket for data exchange: rejected for now because of extra account and token setup. Revisit if Drive sync proves too slow or unreliable.
