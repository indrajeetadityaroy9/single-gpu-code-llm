"""Runs EvalPlus's code generation (evalplus.codegen) against a `vllm serve` endpoint for the SFT and RL models
(paper §6.1), with two changes to its OpenAI backend. The EvalPlus version is the one docker/Dockerfile.eval installs.

- --max_new_tokens (required) reaches the decoder. Unpatched, evalplus.provider.make_model builds OpenAIChatDecoder
  without it, so every request asks for DecoderBase's default of 768 new tokens.
- Each sample's ending is recorded. The requests also ask the server for the generated token ids (`return_token_ids`).
  Every sample adds one row to <samples>.endings.jsonl, next to its samples file and in the same order, with keys:
  - "task_id";
  - "finish_reason": vLLM's, "stop" at the end-of-sequence token (BOS for this model) or "length" at the limit;
  - "closed": the decoded response holds "</think>", where the reasoning parser splits reasoning from answer
    (eval/official_reasoning_parser.py); the paper's finished answers are the closed responses;
  - "loop_stopped": the server's loop stop ended the response (eval/official_loop_stop.py);
  - "tokens": the number of generated tokens, not counting the final BOS.

All arguments are codegen's; scripts/eval_posttrained.sh passes --max_new_tokens.

    OPENAI_BASE_URL=http://<host>:<port>/v1 OPENAI_API_KEY=none python -m eval.official_evalplus <codegen arguments>
"""

import json
import os

import evalplus.codegen
import httpx
import openai
from evalplus.gen.util import openai_request
from evalplus.provider.openai import OpenAIChatDecoder

from corpus.tokenizer import get_tokenizer
from eval.loop import END, stopped

ENDINGS = {}  # "path": the endings file of the samples being generated; "tasks": task id by the message sent
codegen, make_model = evalplus.codegen.codegen, evalplus.codegen.make_model


def sized_make_model(*args, max_new_tokens: int, **kwargs):
    model = make_model(*args, **kwargs)
    model.max_new_tokens = max_new_tokens
    return model


def recording_codegen(target_path: str, model, dataset: dict, *args, **kwargs):
    # the message OpenAIChatDecoder.codegen builds from codegen's prompt, the task's prompt stripped plus "\n"
    ENDINGS.update(path=target_path.removesuffix(".jsonl") + ".endings.jsonl",
                   tasks={model.instruction_prefix + f"\n```python\n{task['prompt'].strip()}\n```": task_id
                          for task_id, task in dataset.items()})
    return codegen(target_path, model, dataset, *args, **kwargs)


def codegen_api_batch(self, message: str, batch_size: int) -> list[str]:
    """OpenAIChatDecoder._codegen_api_batch's request and outputs, the request asking for the generated token ids."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "none"), base_url=self.base_url,
                           http_client=httpx.Client(verify=self.verify_certificate))
    ret = openai_request.make_auto_request(client, message=message, model=self.name, max_tokens=self.max_new_tokens,
                                           temperature=self.temperature, n=batch_size,
                                           extra_body={"return_token_ids": True})
    tokenizer = get_tokenizer()
    bos, decode = tokenizer.bos_token_id, tokenizer.enc.decode
    with open(ENDINGS["path"], "a") as f:
        for choice in ret.choices:
            ids = choice.model_extra["token_ids"]
            f.write(json.dumps({"task_id": ENDINGS["tasks"][message], "finish_reason": choice.finish_reason,
                                "closed": END in decode(ids), "loop_stopped": stopped(ids, bos, decode),
                                "tokens": len(ids) - (ids[-1:] == [bos])}) + "\n")
    return [choice.message.content for choice in ret.choices]


if __name__ == "__main__":
    evalplus.codegen.codegen, evalplus.codegen.make_model = recording_codegen, sized_make_model
    OpenAIChatDecoder._codegen_api_batch = codegen_api_batch
    evalplus.codegen.main()
