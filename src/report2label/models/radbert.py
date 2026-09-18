"""RadBERT-based multi-label abnormality classifier.

Architecture matches the CT-RATE text classifier exactly (RadBERT-RoBERTa-4m
encoder + a single linear head over the pooled output) so that pretrained
`RadBertClassifier.pth` checkpoints load without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class RadBertClassifier(nn.Module):
    def __init__(self, base_model: str = "zzxslp/RadBERT-RoBERTa-4m", n_classes: int = 18):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model)
        self.model = AutoModel.from_pretrained(base_model, config=self.config)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attn_mask)
        return self.classifier(output.pooler_output)
