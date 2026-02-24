from pathlib import Path
from typing import List

import fire
import torch

from generation import MusicLlama
from unconditional_from_scratch import _sanitize_generated_tokens


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _save_tokens(generator: MusicLlama, tokens: List[List[int]], output_path: Path, sanitize_tokens: bool) -> None:
    write_tokens = _sanitize_generated_tokens(tokens, generator.tokenizer, enabled=sanitize_tokens)
    if not write_tokens:
        raise RuntimeError(f"No tokens to write for {output_path}.")
    generator.tokenizer.compound_to_midi(write_tokens).save(str(output_path))


def main(
    ckpt_path: str,
    prompt_midi_path: str,
    output_dir: str = "ab_outputs",
    model_config_path: str = "src/llama_recipes/configs/model_config.json",
    tokenizer_path: str = "recipes/benchmarks/inference_throughput/tokenizer/tokenizer.model",
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: int = 256,
    prompt_len: int = 32,
    base_seed: int = 42,
    sanitize_tokens: bool = True,
    fail_fast_checkpoint_load: bool = True,
    max_allowed_load_issues: int = 8,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this A/B harness.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(base_seed)
    generator = MusicLlama.build(
        ckpt_dir=ckpt_path,
        model_config_path=model_config_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        finetuned_PEFT_weight_path=None,
        seed=base_seed,
        fail_fast=fail_fast_checkpoint_load,
        max_allowed_load_issues=max_allowed_load_issues,
    )

    prompt_raw = generator.tokenizer.midi_to_compound(prompt_midi_path, calibate_to_default_tempo=True)
    prompt_tokens = generator.tokenizer.encode_series(prompt_raw, if_add_sos=True, if_add_eos=False)
    continuation_prompt = prompt_tokens[: max(2, min(len(prompt_tokens), prompt_len))]

    print("[info] A/B harness config")
    print(f"  checkpoint={Path(ckpt_path).resolve()}")
    print(f"  model_config={Path(model_config_path).resolve()}")
    print(f"  prompt_midi={Path(prompt_midi_path).resolve()}")
    print(f"  temperature={temperature} top_p={top_p} max_gen_len={max_gen_len}")
    print(f"  sanitize_tokens={sanitize_tokens} base_seed={base_seed}")

    for i in range(2):
        seed = base_seed + i
        _set_seed(seed)
        scratch_res = generator.music_completion(
            prompt_tokens=[[generator.tokenizer.sos_token_compound]],
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
        )[0]
        scratch_path = out_dir / f"scratch_seed{seed}.mid"
        _save_tokens(generator, scratch_res["generation"]["tokens"], scratch_path, sanitize_tokens)
        print(f"[scratch] seed={seed} tokens={len(scratch_res['generation']['tokens'])} saved={scratch_path.resolve()}")

    for i in range(2):
        seed = base_seed + i
        _set_seed(seed)
        cont_res = generator.music_completion(
            prompt_tokens=[continuation_prompt],
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
        )[0]
        cont_path = out_dir / f"continuation_seed{seed}.mid"
        _save_tokens(generator, cont_res["generation"]["tokens"], cont_path, sanitize_tokens)
        print(f"[continuation] seed={seed} prompt_len={len(continuation_prompt)} tokens={len(cont_res['generation']['tokens'])} saved={cont_path.resolve()}")


if __name__ == "__main__":
    fire.Fire(main)
