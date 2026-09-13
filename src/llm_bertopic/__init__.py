"""Research pipeline; importing this package never downloads models or data."""

import os

# This project exclusively uses PyTorch. Avoid optional TensorFlow/Keras import conflicts.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.3.1"
