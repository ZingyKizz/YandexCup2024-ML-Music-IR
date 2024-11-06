import timm
import torch
from torch import nn

from csi.base.model.pooling import global_avg_pooling_2d

IN_CHANNELS = 3


class DebugModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(84)
        self.fc0 = nn.Linear(84, 8)
        self.hswish0 = nn.Hardswish()
        self.fc1 = nn.Linear(8, 8)
        self.fc = nn.Linear(8, 41617)
        self.hswish1 = nn.Hardswish()

    def forward(self, input):
        emb = self._calc_embed(input["cqt"])
        if input["pos_cqt"].size(1) != 0:
            pos_emb = self._calc_embed(input["pos_cqt"])
        else:
            pos_emb = torch.empty(0)
        clique_logits = self.fc(self.hswish1(emb))
        track_id = input["track_id"]
        return {
            "embedding": emb,
            "pos_embedding": pos_emb,
            "clique_logits": clique_logits,
            "track_id": track_id,
        }

    def _calc_embed(self, cqt):
        x = self.ln(cqt)
        x = x.mean(dim=1)
        x = self.fc0(x)
        x = self.hswish0(x)
        emb = self.fc1(x)
        return emb


class TimmModel(nn.Module):
    def __init__(
        self,
        backbone_name,
        layer_norm_size,
        emb_size=512,
        num_classes=41617,
        calc_pos_embeddings=True,
    ):
        super().__init__()
        self._init_backbone(
            backbone_name, in_chans=IN_CHANNELS, num_classes=num_classes, pretrained=True
        )
        self.fc_emb = nn.Linear(self.model.num_features, emb_size)
        self.clf = nn.Linear(emb_size, num_classes)
        self._init_ln(layer_norm_size)
        self.calc_pos_embeddings = calc_pos_embeddings

    def forward(self, input):
        emb = self._get_embed(input["cqt"])

        if (input["pos_cqt"].size(1) != 0) and self.calc_pos_embeddings:
            pos_emb = self._get_embed(input["pos_cqt"])
        else:
            pos_emb = torch.empty(0)

        clique_logits = self.clf(emb)

        return {
            "embedding": emb,
            "pos_embedding": pos_emb,
            "clique_logits": clique_logits,
            "track_id": input["track_id"],
        }

    def _get_embed(self, x):
        x = self.ln(x)
        x = self.model.forward_features(x)
        x = global_avg_pooling_2d(x)
        return self.fc_emb(x)

    def _init_ln(self, layer_norm_size):
        self.ln = nn.LayerNorm(layer_norm_size)
        self.ln.bias.data.zero_()
        self.ln.weight.data.fill_(1.0)

    def _init_backbone(self, *args, **kwargs):
        self.model = timm.create_model(*args, **kwargs)
        self.model.reset_classifier(0)
