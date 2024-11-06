import torch
from torch import nn


class MetricLearner(nn.Module):
    def __init__(self, loss_fn, miner=None, loss_optimizer_cls=None, **loss_optimizer_args):
        super().__init__()
        self.loss_fn = loss_fn
        self.miner = miner
        if isinstance(loss_optimizer_cls, str):
            loss_optimizer_cls = getattr(torch.optim, loss_optimizer_cls)
            self.loss_optimizer = loss_optimizer_cls(
                self.loss_fn.parameters(), **loss_optimizer_args
            )

    def forward(self, model, out, batch):
        if batch["clique"].dim() > 1 and batch["clique"].size(-1) != 1:
            return torch.tensor(0.0, requires_grad=True).to(out["embedding"].device)
        if self.miner is not None:
            embeddings = torch.cat((out["embedding"], out["pos_embedding"]), dim=0)
            if "og_clique" in batch:
                labels = torch.cat((batch["og_clique"], batch["og_clique"]), dim=0)
            else:
                labels = torch.cat((batch["clique"], batch["clique"]), dim=0)
            loss = self.loss_fn(embeddings, labels, self.miner(embeddings, labels))
        else:
            loss = self.loss_fn(out["embedding"], batch["clique"])
        return loss


class PairCrossEntropy(nn.Module):
    def __init__(self, normalize=True, **ce_params):
        super().__init__()
        self.fn = nn.CrossEntropyLoss(**ce_params)
        self.scale = nn.Parameter(torch.ones(1))
        self.normalize = normalize

    def __call__(self, embeddings, labels, miner_output):
        device = embeddings.device
        anc1_indices, pos_indices, anc2_indices, neg_indices = miner_output

        logits_pos = (embeddings[anc1_indices] * embeddings[pos_indices] * self.scale).sum(dim=1)
        if self.normalize:
            logits_pos /= torch.norm(embeddings[anc1_indices], p=2, dim=1) * torch.norm(
                embeddings[pos_indices], p=2, dim=1
            )

        logits_neg = (embeddings[anc2_indices] * embeddings[neg_indices] * self.scale).sum(dim=1)
        if self.normalize:
            logits_neg /= torch.norm(embeddings[anc2_indices], p=2, dim=1) * torch.norm(
                embeddings[neg_indices], p=2, dim=1
            )

        logits = torch.cat((logits_pos, logits_neg), dim=0)
        labels = torch.cat(
            (
                torch.ones(logits_pos.size(0)),
                torch.zeros(logits_neg.size(0)),
            )
        ).to(device)

        loss = self.fn(logits, labels)
        return loss


class CosineEmbedding(nn.Module):
    def __init__(self, **ce_params):
        super().__init__()
        self.fn = nn.CosineEmbeddingLoss(**ce_params)

    def __call__(self, embeddings, labels, miner_output):
        device = embeddings.device
        anc1_indices, pos_indices, anc2_indices, neg_indices = miner_output

        pos_labels = torch.ones(anc1_indices.size(0)).to(device)
        loss_pos = self.fn(embeddings[anc1_indices], embeddings[pos_indices], pos_labels)

        neg_labels = -1 * torch.ones(anc2_indices.size(0)).to(device)
        loss_neg = self.fn(embeddings[anc2_indices], embeddings[neg_indices], neg_labels)

        loss = loss_pos + loss_neg
        return loss


class IPairCrossEntropy(nn.Module):
    def __init__(self, **ce_params):
        super().__init__()
        self.fn = nn.CrossEntropyLoss(**ce_params)
        self.scale = nn.Parameter(torch.ones(1))

    def __call__(self, embeddings, labels, miner_output):
        device = embeddings.device
        anc1_indices, pos_indices, anc2_indices, neg_indices = miner_output

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        scale = self.scale.exp()
        logits_pos = (embeddings[anc1_indices] * embeddings[pos_indices] * scale).sum(dim=1)
        logits_neg = (embeddings[anc2_indices] * embeddings[neg_indices] * scale).sum(dim=1)

        logits = torch.cat((logits_pos, logits_neg), dim=0)
        labels = torch.cat(
            (
                torch.ones(logits_pos.size(0)),
                torch.zeros(logits_neg.size(0)),
            )
        ).to(device)

        loss = self.fn(logits, labels)
        return loss
