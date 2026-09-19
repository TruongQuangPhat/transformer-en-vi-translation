from pathlib import Path
from math import pow

import torch

from data.tokenizer import load_tokenizers
from model.transformer import Transformer


D_MODEL = 128
NUM_HEADS = 4
D_FF = 512

NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

DROPOUT = 0.1
MAX_LEN = 512

CHECKPOINT_PATH = Path("checkpoints/best.pt")


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_pad_id: int,
    tgt_pad_id: int,
) -> Transformer:
    """Load a trained Transformer from a checkpoint."""

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_pad_id=src_pad_id,
        tgt_pad_id=tgt_pad_id,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def greedy_decode(
    model: Transformer,
    src_ids: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
) -> torch.Tensor:
    """
    Generate target tokens using greedy decoding.

    Args:
        model: Trained Transformer model.
        src_ids: Source token IDs of shape [1, S_src].
        bos_id: Beginning-of-sentence token ID.
        eos_id: End-of-sentence token ID.
        max_len: Maximum target sequence length.

    Returns:
        Generated target token IDs of shape [1, S_tgt].
    """
    model.eval()

    with torch.no_grad():
        memory, _ = model.encode(src_ids)

        generated_ids = torch.tensor(
            [[bos_id]],
            dtype=torch.long,
            device=src_ids.device,
        )

        for _ in range(max_len - 1):
            logits, _, _ = model.decode(
                tgt_ids=generated_ids,
                memory=memory,
                src_ids=src_ids,
            )

            next_token = logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )

            generated_ids = torch.cat(
                [generated_ids, next_token],
                dim=1,
            )

            if next_token.item() == eos_id:
                break

    return generated_ids

def beam_search_decode(
    model: Transformer,
    src_ids: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
    beam_size: int = 5,
    length_penalty: float = 0.6,
) -> torch.Tensor:
    """
    Generate target tokens using beam search.

    Args:
        model: Trained Transformer model.
        src_ids: Source token IDs of shape [1, S_src].
        bos_id: Beginning-of-sentence token ID.
        eos_id: End-of-sentence token ID.
        max_len: Maximum target sequence length.
        beam_size: Number of candidate sequences kept at each step.
        length_penalty: Strength of length normalization.

    Returns:
        Best generated target token IDs of shape [1, S_tgt].
    """
    model.eval()

    with torch.no_grad():
        memory, _ = model.encode(src_ids)

        # Each beam is:
        # (token_ids, cumulative_log_probability, finished)
        beams = [
            (
                [bos_id],
                0.0,
                False,
            )
        ]

        for _ in range(max_len - 1):
            candidates = []

            for token_ids, log_prob, finished in beams:
                # Do not expand sequences that already reached EOS.
                if finished:
                    candidates.append(
                        (
                            token_ids,
                            log_prob,
                            True,
                        )
                    )
                    continue

                tgt_ids = torch.tensor(
                    [token_ids],
                    dtype=torch.long,
                    device=src_ids.device,
                )

                logits, _, _ = model.decode(
                    tgt_ids=tgt_ids,
                    memory=memory,
                    src_ids=src_ids,
                )

                # Only the last timestep predicts the next token.
                next_token_logits = logits[:, -1, :]

                log_probs = torch.log_softmax(
                    next_token_logits,
                    dim=-1,
                )

                top_log_probs, top_token_ids = torch.topk(
                    log_probs,
                    k=beam_size,
                    dim=-1,
                )

                for next_log_prob, next_token_id in zip(
                    top_log_probs[0],
                    top_token_ids[0],
                ):
                    next_token_id = next_token_id.item()
                    next_log_prob = next_log_prob.item()

                    new_token_ids = token_ids + [
                        next_token_id
                    ]

                    new_log_prob = (
                        log_prob + next_log_prob
                    )

                    candidates.append(
                        (
                            new_token_ids,
                            new_log_prob,
                            next_token_id == eos_id,
                        )
                    )

            def normalized_score(
                beam: tuple[list[int], float, bool],
            ) -> float:
                token_ids, log_prob, _ = beam

                length = len(token_ids)

                penalty = pow(
                    (5.0 + length) / 6.0,
                    length_penalty,
                )

                return log_prob / penalty

            candidates.sort(
                key=normalized_score,
                reverse=True,
            )

            beams = candidates[:beam_size]

            if all(
                finished
                for _, _, finished in beams
            ):
                break

        best_beam = max(
            beams,
            key=normalized_score,
        )

        best_token_ids = best_beam[0]

        return torch.tensor(
            [best_token_ids],
            dtype=torch.long,
            device=src_ids.device,
        )


def translate_sentence(
    model: Transformer,
    src_tokenizer,
    tgt_tokenizer,
    text: str,
    device: torch.device,
    max_len: int = 128,
    beam_size: int = 5,
) -> str:
    """
    Translate one source sentence using greedy decoding.
    """

    src_ids = src_tokenizer.encode_ids(
        text,
        add_bos=True,
        add_eos=True,
    )

    src_tensor = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=device,
    )

    if beam_size == 1:
        generated_ids = greedy_decode(
            model=model,
            src_ids=src_tensor,
            bos_id=tgt_tokenizer.bos_id(),
            eos_id=tgt_tokenizer.eos_id(),
            max_len=max_len,
        )
    else:
        generated_ids = beam_search_decode(
            model=model,
            src_ids=src_tensor,
            bos_id=tgt_tokenizer.bos_id(),
            eos_id=tgt_tokenizer.eos_id(),
            max_len=max_len,
            beam_size=beam_size,
        )

    generated_ids = generated_ids.squeeze(0).tolist()

    return tgt_tokenizer.decode(generated_ids)


def main() -> None:
    """Run a simple translation example."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    en_tokenizer, vi_tokenizer = load_tokenizers()

    model = load_model(
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
        src_vocab_size=en_tokenizer.vocab_size(),
        tgt_vocab_size=vi_tokenizer.vocab_size(),
        src_pad_id=en_tokenizer.pad_id(),
        tgt_pad_id=vi_tokenizer.pad_id(),
    )

    source_sentences = [
        "I love machine learning.",
        "I am a student.",
        "This is a book.",
        "How are you?",
        "I go to school every day.",
    ]

    for source_text in source_sentences:
        greedy_translation = translate_sentence(
            model=model,
            src_tokenizer=en_tokenizer,
            tgt_tokenizer=vi_tokenizer,
            text=source_text,
            device=device,
            beam_size=1,
        )

        beam_translation = translate_sentence(
            model=model,
            src_tokenizer=en_tokenizer,
            tgt_tokenizer=vi_tokenizer,
            text=source_text,
            device=device,
            beam_size=5,
        )

        print("Source :", source_text)
        print("Greedy :", greedy_translation)
        print("Beam   :", beam_translation)
        print()


if __name__ == "__main__":
    main()