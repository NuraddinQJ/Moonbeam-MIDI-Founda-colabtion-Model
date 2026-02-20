from pathlib import Path
from typing import Optional

import fire
import torch

from generation import MusicLlama


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
) -> None:
    """Generate a MIDI file from an SOS-only prompt (no dataset required)."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = MusicLlama.build(
        ckpt_dir=ckpt_path,
        model_config_path=model_config_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        finetuned_PEFT_weight_path=finetuned_PEFT_weight_path,
        seed=seed,
    )

    sos_prompt = [generator.tokenizer.sos_token_compound]
    result = generator.music_completion(
        prompt_tokens=[sos_prompt],
        temperature=temperature,
        top_p=top_p,
        max_gen_len=max_gen_len,
    )[0]

    output_path = Path(output_midi_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["generation"]["content"].save(str(output_path))
    print(f"Saved MIDI to: {output_path.resolve()}")


if __name__ == "__main__":
    fire.Fire(main)
