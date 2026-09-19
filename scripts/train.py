from pathlib import Path

import torch

from data.dataset import (
    TranslationDataset,
    create_dataloader,
)

from data.tokenizer import load_tokenizers
from model.transformer import Transformer
from training.loss import TranslationLoss
from training.optimizer import (
    create_noam_scheduler,
    create_optimizer,
)
from training.trainer import Trainer


TRAIN_PATH = Path("data/processed/train.jsonl")
VAL_PATH = Path("data/processed/validation.jsonl")

BATCH_SIZE = 32

D_MODEL = 128
NUM_HEADS = 4
D_FF = 512

NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

DROPOUT = 0.1
MAX_LEN = 512

NUM_EPOCHS = 30

WARMUP_STEPS = 8000
LR_SCALE = 1.5

CHECKPOINT_DIR = Path("checkpoints")
RESUME = False
CHECKPOINT_PATH = CHECKPOINT_DIR / "latest.pt"


def main() -> None:
    """Train the Transformer translation model."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    en_tokenizer, vi_tokenizer = load_tokenizers()

    train_dataset = TranslationDataset(
        jsonl_path=TRAIN_PATH,
        src_tokenizer=en_tokenizer,
        tgt_tokenizer=vi_tokenizer,
    )

    val_dataset = TranslationDataset(
        jsonl_path=VAL_PATH,
        src_tokenizer=en_tokenizer,
        tgt_tokenizer=vi_tokenizer,
    )

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        src_pad_id=en_tokenizer.pad_id(),
        tgt_pad_id=vi_tokenizer.pad_id(),
        shuffle=True,
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        src_pad_id=en_tokenizer.pad_id(),
        tgt_pad_id=vi_tokenizer.pad_id(),
        shuffle=False,
    )

    model = Transformer(
        src_vocab_size=en_tokenizer.vocab_size(),
        tgt_vocab_size=vi_tokenizer.vocab_size(),
        src_pad_id=en_tokenizer.pad_id(),
        tgt_pad_id=vi_tokenizer.pad_id(),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(device)

    criterion = TranslationLoss(
        pad_id=vi_tokenizer.pad_id(),
    )

    optimizer = create_optimizer(
        model.parameters(),
    )

    scheduler = create_noam_scheduler(
        optimizer=optimizer,
        d_model=D_MODEL,
        warmup_steps=WARMUP_STEPS,
        scale=LR_SCALE,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    start_epoch = 1
    best_val_loss = float("inf")

    if RESUME:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT_PATH}"
            )

        start_epoch, best_val_loss = trainer.load_checkpoint(
            CHECKPOINT_PATH,
        )

    trainer.fit(
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
    )


if __name__ == "__main__":
    main()