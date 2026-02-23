from pathlib import Path
from typing import Optional

import fire
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from generation import MusicLlama
from llama_recipes.datasets.music_tokenizer import MusicTokenizer


def _normalize_checkpoint_state_dict(checkpoint: dict) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dict-like object.")

    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    normalized = {}
    for k, v in state_dict.items():
        normalized[k[7:] if k.startswith("module.") else k] = v
    return normalized


def _load_checkpoint_state_dict(ckpt_path: str) -> dict:
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    return _normalize_checkpoint_state_dict(checkpoint)


def _resolve_model_config_path(ckpt_path: str, model_config_path: str) -> str:
    model_config = Path(model_config_path)
    small_config = model_config.with_name("model_config_small.json")
    if not small_config.exists():
        return str(model_config)

    state_dict = _load_checkpoint_state_dict(ckpt_path)
    ckpt_sig = MusicLlama.checkpoint_signature(state_dict)

    cfg = LlamaConfig.from_pretrained(str(model_config))
    cfg_small = LlamaConfig.from_pretrained(str(small_config))
    cfg_sig = MusicLlama.config_signature(cfg)
    small_sig = MusicLlama.config_signature(cfg_small)

    if all(
        ckpt_sig.get(k) is not None and ckpt_sig.get(k) == small_sig.get(k)
        for k in ("hidden_size", "num_hidden_layers", "vocab_size")
    ) and any(
        ckpt_sig.get(k) is not None and ckpt_sig.get(k) != cfg_sig.get(k)
        for k in ("hidden_size", "num_hidden_layers", "vocab_size")
    ):
        print(f"[info] Switching model config to checkpoint-compatible file: {small_config}")
        return str(small_config)

    return str(model_config)


def _build_music_llama(
    ckpt_path: str,
    model_config_path: str,
    seed: int,
    fail_fast: bool,
    max_allowed_load_issues: int,
):
    del seed
    model_config_path = _resolve_model_config_path(ckpt_path, model_config_path)
    config = LlamaConfig.from_pretrained(model_config_path)
    model = LlamaForCausalLM(config)

    state_dict = _load_checkpoint_state_dict(ckpt_path)
    checkpoint_sig = MusicLlama.checkpoint_signature(state_dict)
    config_sig = MusicLlama.config_signature(config)

    filtered_state, skipped_shape_mismatch = MusicLlama._filter_state_dict_for_model(model, state_dict)
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    load_summary = MusicLlama.summarize_checkpoint_load(
        state_dict=state_dict,
        filtered_state_dict=filtered_state,
        skipped_shape_mismatch=skipped_shape_mismatch,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )

    MusicLlama.validate_checkpoint_load(
        summary=load_summary,
        checkpoint_sig=checkpoint_sig,
        config_sig=config_sig,
        max_allowed_issues=max_allowed_load_issues,
        fail_fast=fail_fast,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    tokenizer = MusicTokenizer(
        timeshift_vocab_size=config.onset_vocab_size,
        dur_vocab_size=config.dur_vocab_size,
        octave_vocab_size=config.octave_vocab_size,
        pitch_class_vocab_size=config.pitch_class_vocab_size,
        instrument_vocab_size=config.instrument_vocab_size,
        velocity_vocab_size=config.velocity_vocab_size,
    )

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)

    return MusicLlama(model, tokenizer, config), load_summary, model_config_path, checkpoint_sig, config_sig


def _sanitize_generated_tokens(tokens, tokenizer, enabled: bool = True):
    """Minimal sanitization: keep valid 6-tuples and clamp indices into valid range."""
    if not enabled:
        return tokens

    cleaned = []
    bounds = [
        tokenizer.timeshift_vocab_size,
        tokenizer.dur_vocab_size,
        tokenizer.octave_vocab_size,
        tokenizer.pitch_class_vocab_size,
        tokenizer.instrument_vocab_size,
        tokenizer.velocity_vocab_size,
    ]

    for row in tokens:
        if len(row) != 6:
            continue
        cleaned.append([
            min(max(int(value), 0), vocab_size - 1) for value, vocab_size in zip(row, bounds)
        ])
    return cleaned


def main(
    ckpt_path: str,
    model_config_path: str = "src/llama_recipes/configs/model_config.json",
    tokenizer_path: str = "recipes/benchmarks/inference_throughput/tokenizer/tokenizer.model",
    output_midi_path: str = "out.mid",
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_gen_len: int = 256,
    seed: int = 42,
    finetuned_PEFT_weight_path: Optional[str] = None,
    num_samples: int = 1,
    sanitize_tokens: bool = True,
    fail_fast_checkpoint_load: bool = True,
    max_allowed_load_issues: int = 8,
) -> None:
    """Generate MIDI from SOS-only prompt with strict checkpoint-load validation."""
    del tokenizer_path, max_seq_len, finetuned_PEFT_weight_path

    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this Colab quickstart. In Colab, set Runtime -> Change runtime type -> GPU.")
    torch.cuda.manual_seed_all(seed)

    generator, load_summary, resolved_config_path, checkpoint_sig, config_sig = _build_music_llama(
        ckpt_path=ckpt_path,
        model_config_path=model_config_path,
        seed=seed,
        fail_fast=fail_fast_checkpoint_load,
        max_allowed_load_issues=max_allowed_load_issues,
    )

    print("[diagnostic] checkpoint_load_summary")
    print(f"checkpoint_total_keys={load_summary['total_checkpoint_keys']}")
    print(f"loaded_keys={load_summary['loaded_keys']}")
    print(f"skipped_shape_mismatch={load_summary['skipped_shape_mismatch']}")
    print(f"missing_keys={load_summary['missing_keys']}")
    print(f"unexpected_keys={load_summary['unexpected_keys']}")

    print("[diagnostic] checkpoint_architecture")
    print(f"hidden_size={checkpoint_sig.get('hidden_size')}")
    print(f"num_layers={checkpoint_sig.get('num_hidden_layers')}")
    print(f"vocab_size={checkpoint_sig.get('vocab_size')}")

    print("[diagnostic] resolved_model_config")
    print(f"config_file_used={Path(resolved_config_path).resolve()}")
    print(f"hidden_size={config_sig.get('hidden_size')}")
    print(f"num_layers={config_sig.get('num_hidden_layers')}")
    print(f"vocab_size={config_sig.get('vocab_size')}")

    print("[diagnostic] generation_parameters")
    print(f"temperature={temperature}")
    print(f"top_p={top_p}")
    print(f"max_gen_len={max_gen_len}")
    print(f"seed={seed}")

    print("[diagnostic] sanitization")
    print(f"sanitization_enabled={sanitize_tokens}")

    num_samples = max(1, int(num_samples))
    output_path = Path(output_midi_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        sample_seed = seed + i
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)

        sos_prompt = [generator.tokenizer.sos_token_compound]
        result = generator.music_completion(
            prompt_tokens=[sos_prompt],
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
        )[0]

        output_tokens = _sanitize_generated_tokens(
            result["generation"]["tokens"],
            generator.tokenizer,
            enabled=sanitize_tokens,
        )
        if not output_tokens:
            raise RuntimeError(f"No valid generated tokens remained for sample {i}.")

        sample_path = output_path if num_samples == 1 else output_path.with_name(f"{output_path.stem}_{i+1}{output_path.suffix}")
        generator.tokenizer.compound_to_midi(output_tokens).save(str(sample_path))
        print(
            f"Saved MIDI to: {sample_path.resolve()} | generated_tokens={len(result['generation']['tokens'])} "
            f"written_tokens={len(output_tokens)} | seed={sample_seed}"
        )


if __name__ == "__main__":
    fire.Fire(main)
