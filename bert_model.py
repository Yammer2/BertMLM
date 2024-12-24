import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertModel, BertConfig, BertForMaskedLM

class LightningBert(L.LightningModule):

    def __init__(self, config: BertConfig, sequence_length: int, input_dim: int, num_mask: int):
        super().__init__()
        self.model = BertModel(config)
        self.regression_head = nn.Linear(config.hidden_size, input_dim)

        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.num_mask = num_mask

    def forward(self, inputs_embeds, attention_mask, labels=None):

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        sequence_output = outputs.last_hidden_state
        predictions = self.regression_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss(reduction="mean")
            mask = labels != -100
            mask = mask.all(dim=-1)
            loss = loss_fn(predictions[mask], labels[mask])

        return {"loss": loss, "predictions": predictions}

    def training_step(self, batch, batch_idx):
        inputs = {
            "inputs_embeds": batch,
            "attention_mask": torch.ones(batch.size(0), self.sequence_length).to(self.device),
            "labels": self.generate_mlm_labels(batch)
        }

        outputs = self(**inputs)

        self.log("train/total_loss", outputs["loss"], prog_bar=True)
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        inputs = {
            "inputs_embeds": batch,
            "attention_mask": torch.ones(batch.size(0), self.sequence_length).to(self.device),
            "labels": self.generate_mlm_labels(batch)
        }

        outputs = self(**inputs)
        self.log("val/total_loss", outputs["loss"])
        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        inputs = {
            "inputs_embeds": batch,
            "attention_mask": torch.ones(batch.size(0), self.sequence_length).to(self.device),
            "labels": self.generate_mlm_labels(batch)
        }

        outputs = self(**inputs)
        self.log("test/total_loss", outputs["loss"])
        return outputs["loss"]

    def generate_mlm_labels(self, inputs_embeds):
        size = inputs_embeds.size(0)
        # print(inputs_embeds.shape)
        mlm_labels = torch.full((size, self.sequence_length, self.input_dim), -100, dtype=torch.float).to(self.device)
        mask_positions = torch.randint(0, self.sequence_length, (size, self.num_mask))

        for i in range(size):
            mlm_labels[i, mask_positions[i], :] = inputs_embeds[i, mask_positions[i], :]

        return mlm_labels.to(self.device)


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)
