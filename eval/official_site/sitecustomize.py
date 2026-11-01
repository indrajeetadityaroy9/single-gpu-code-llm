"""On PYTHONPATH in the vLLM containers: every process registers the "sgc" reasoning parser
(eval/official_reasoning_parser.py), because `vllm serve --reasoning-parser-plugin` imports it in the API server only
and the spawned engine core looks the parser up too."""
import eval.official_reasoning_parser  # noqa: F401 (registers the parser)
