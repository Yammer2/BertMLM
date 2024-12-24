import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForMaskedLM
from torch.optim import AdamW
from dataset import TimeSeriesDataset, get_data_loaders
from bert_model import LightningBert
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers.wandb import WandbLogger
from setproctitle import setproctitle
from argparse import ArgumentParser

def main(time_step, feature_dim, batch_size, num_attention_heads, num_mask):
    # Create dataset and dataloader
    train_loader, val_loader, test_loader = get_data_loaders(
        file_path="synthetic_sine_cosine_series.csv",
        time_steps=time_step,
        dimensions=feature_dim,
        batch_size=batch_size,
    )

    # Define BERT model configuration
    config = BertConfig(
        vocab_size=1,  # Typically required for MLM
        hidden_size=feature_dim,
        num_hidden_layers=12,
        num_attention_heads=num_attention_heads,
        intermediate_size=3072,
        max_position_embeddings=128,
    )

    bert = LightningBert(config, time_step, feature_dim, num_mask)

    model_ckpt = ModelCheckpoint(
        dirpath="ckpt/",
        save_top_k=1,
        monitor="val/total_loss",
        mode="min"
    )

    early_stopping = EarlyStopping(
        monitor="val/total_loss",
        patience=5,
        mode="min"
    )

    logger = WandbLogger(name="bert", project="BERT pretraining", offline=False)

    trainer = Trainer(
        accelerator="cuda",
        devices=[1],
        deterministic=True,
        callbacks=[model_ckpt, early_stopping],
        logger=logger,
        max_epochs=100,
        # benchmark=True
    )

    trainer.fit(bert, train_loader, val_loader)
    print("best model path", trainer.checkpoint_callback.best_model_path)

    trainer.test(bert, test_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--time_step", type=int, default=40, help="時系列長")
    parser.add_argument("--feature_dim", type=int, default=2, help="次元数")
    parser.add_argument("--batch_size", type=int, default=256, help="バッチサイズ")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="Attention Head数 (次元数で割り切れる数)")
    parser.add_argument("--num_mask", type=int, default=10, help="マスクするトークン数")
    args = parser.parse_args()

    main(args.time_step, args.feature_dim, args.batch_size, args.num_attention_heads, args.num_mask)
