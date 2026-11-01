"""
The SFT data's name, its single training stage and the chat format's reasoning markers (paper §5.2.1). Imports
neither torch nor the data, so every process that plans or runs SFT can use it; the build is corpus/posttrain/build.py.
"""
from corpus.mixture import REPO_LEVEL_STAGE, Group, Stage

# The data (corpus/layout.py paths) and its stage: sequences of 32,768 tokens, 3 per micro-batch, as in the
# repository-level pretraining stage; tokens 0, since no token budget sizes its packing (mixture.Stage.rows)
NAME = "sft"
SFT_STAGE = Stage(REPO_LEVEL_STAGE.seq_len, 0, {NAME: Group(1.0, (NAME,))})
# The reasoning's tags; the context ends with THINK; END_THINK separates the reasoning from the answer
OPEN, CLOSE = "<think>", "</think>"
THINK, END_THINK = f"\n\n{OPEN}\n", f"\n{CLOSE}\n\n"
