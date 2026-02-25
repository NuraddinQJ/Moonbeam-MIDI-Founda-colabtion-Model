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


def _resolve_model_config_path(ckpt_path: str, model_config_path: str) -> str:
    model_config = Path(model_config_path)
    small_config = model_config.with_name("model_config_small.json")
    if not small_config.exists():
        return str(model_config)

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    state_dict = _normalize_checkpoint_state_dict(checkpoint)
    ckpt_hidden_size = state_dict.get("model.norm.weight").shape[0] if "model.norm.weight" in state_dict else None
    if ckpt_hidden_size is None:
        return str(model_config)

    cfg_hidden = LlamaConfig.from_pretrained(str(model_config)).hidden_size
    small_hidden = LlamaConfig.from_pretrained(str(small_config)).hidden_size
    if ckpt_hidden_size == small_hidden and ckpt_hidden_size != cfg_hidden:
        print(f"[info] Switching model config to checkpoint-compatible file: {small_config}")
        return str(small_config)
    return str(model_config)


def _build_music_llama(
    ckpt_path: str,
    model_config_path: str,
    seed: int,
) -> MusicLlama:
    model_config_path = _resolve_model_config_path(ckpt_path, model_config_path)
    config = LlamaConfig.from_pretrained(model_config_path)
    model = LlamaForCausalLM(config)

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    state_dict = _normalize_checkpoint_state_dict(checkpoint)
    model_state = model.state_dict()
    filtered_state = {
        k: v for k, v in state_dict.items() if k in model_state and getattr(v, "shape", None) == model_state[k].shape
    }
    skipped = len(state_dict) - len(filtered_state)
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(f"[info] Loaded keys: {len(filtered_state)} | skipped: {skipped} | missing: {len(missing)} | unexpected: {len(unexpected)}")

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

    return MusicLlama(model, tokenizer, config)




def _sanitize_generated_tokens(tokens, tokenizer):
    """Repair generated token rows for MIDI safety while preserving as much content as possible."""
    cleaned = []
    last_onset = 0
    max_octave = max(0, tokenizer.octave_vocab_size - 3)
    max_pitch = max(0, tokenizer.pitch_class_vocab_size - 3)
    max_instrument = max(0, tokenizer.instrument_vocab_size - 3)
    max_velocity = max(0, tokenizer.velocity_vocab_size - 3)

    for row in tokens:
        if len(row) != 6:
            continue
        onset, duration, octave, pitch, instrument, velocity = [int(x) for x in row]

        onset = max(0, onset)
        duration = max(1, duration)
        onset = max(onset, last_onset)

        octave = min(max(0, octave), max_octave)
        pitch = min(max(0, pitch), max_pitch)
        instrument = min(max(0, instrument), max_instrument)
        velocity = min(max(0, velocity), max_velocity)

        cleaned.append([onset, duration, octave, pitch, instrument, velocity])
        last_onset = onset
    return cleaned

def main(
    ckpt_path: str,
    model_config_path: str = "src/llama_recipes/configs/model_config.json",
    tokenizer_path: str = "recipes/benchmarks/inference_throughput/tokenizer/tokenizer.model",
    output_midi_path: str = "out.mid",
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_gen_len: int = 512,
    seed: int = 42,
    finetuned_PEFT_weight_path: Optional[str] = None,
    num_samples: int = 1,
    min_generated_tokens: int = 32,
    max_attempts_per_sample: int = 4,
) -> None:
    """Generate a MIDI file from an SOS-only prompt (no dataset required)."""
    del tokenizer_path, max_seq_len, finetuned_PEFT_weight_path

    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this Colab quickstart. In Colab, set Runtime -> Change runtime type -> GPU.")
    torch.cuda.manual_seed_all(seed)

    generator = _build_music_llama(
        ckpt_path=ckpt_path,
        model_config_path=model_config_path,
        seed=seed,
    )

    num_samples = max(1, int(num_samples))
    output_path = Path(output_midi_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        sanitized_tokens = []
        used_seed = None
        used_temperature = None

        for attempt in range(max(1, int(max_attempts_per_sample))):
            sample_seed = seed + i + attempt
            attempt_temperature = min(1.3, float(temperature) + 0.05 * attempt)

            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)

            sos_prompt = [generator.tokenizer.sos_token_compound]
            result = generator.music_completion(
                prompt_tokens=[sos_prompt],
                temperature=attempt_temperature,
                top_p=top_p,
                max_gen_len=max_gen_len,
            )[0]

            sanitized_tokens = _sanitize_generated_tokens(result["generation"]["tokens"], generator.tokenizer)
            if len(sanitized_tokens) >= max(1, int(min_generated_tokens)):
                used_seed = sample_seed
                used_temperature = attempt_temperature
                break

            print(
                f"[warn] Sample {i+1}: attempt {attempt+1}/{max_attempts_per_sample} was too short "
                f"({len(sanitized_tokens)} tokens). Retrying with a new seed/temperature."
            )

        if len(sanitized_tokens) < max(1, int(min_generated_tokens)):
            raise RuntimeError(
                f"Sample {i+1} stayed below min_generated_tokens={min_generated_tokens} after "
                f"{max_attempts_per_sample} attempts."
            )

        sample_path = output_path if num_samples == 1 else output_path.with_name(f"{output_path.stem}_{i+1}{output_path.suffix}")
        generator.tokenizer.compound_to_midi(sanitized_tokens).save(str(sample_path))
        print(
            f"Saved MIDI to: {sample_path.resolve()} | sanitized_tokens={len(sanitized_tokens)} "
            f"| seed={used_seed} | temperature={used_temperature:.2f}"
        )


if __name__ == "__main__":
    fire.Fire(main)
