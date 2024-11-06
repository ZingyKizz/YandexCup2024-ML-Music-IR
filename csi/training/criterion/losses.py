import torch
from torch import nn
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    def __init__(self, label_smoothing=0.0, **params):
        super().__init__()
        self.fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, model, out, batch):
        return self.fn(out["clique_logits"], batch["clique"])


class MSE(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.fn = nn.MSELoss()

    def forward(self, model, out, batch):
        return self.fn(
            out["num_other_tracks_in_clique_pred"].float(),
            batch["num_other_tracks_in_clique"].float(),
        )


class Huber(nn.Module):
    def __init__(self, delta=1.0, **params):
        super().__init__()
        self.fn = nn.HuberLoss(delta=delta)

    def forward(self, model, out, batch):
        return self.fn(
            out["num_other_tracks_in_clique_pred"].float(),
            batch["num_other_tracks_in_clique"].float(),
        )


class LogHuber(nn.Module):
    def __init__(self, delta=1.0, **params):
        super().__init__()
        self.fn = nn.HuberLoss(delta=delta)

    def forward(self, model, out, batch):
        return self.fn(
            out["num_other_tracks_in_clique_pred"].float(),
            torch.log(batch["num_other_tracks_in_clique"].float()),
        )


class BCEFocal(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2.0, **params):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, model, out, batch):
        preds = out["clique_logits"]
        targets = batch["clique"]
        if targets.size(-1) != self.num_classes:
            targets = F.one_hot(targets, self.num_classes).float()
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = (
            targets * self.alpha * (1.0 - probas) ** self.gamma * bce_loss
            + (1.0 - targets) * probas**self.gamma * bce_loss
        )
        loss = loss.mean()
        return loss


class Focal(nn.Module):
    def __init__(self, alpha=None, gamma=2, ignore_index=-100, reduction='mean'):
        super().__init__()
        # use standard CE loss without reducion as basis
        self.CE = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, model, out, batch):
        '''
        input (B, N)
        target (B)
        '''
        preds = out["clique_logits"]
        targets = batch["clique"]
        minus_logpt = self.CE(preds, targets)
        pt = torch.exp(-minus_logpt)  # don't forget the minus here
        focal_loss = (1 - pt) ** self.gamma * minus_logpt

        # apply class weights
        if self.alpha != None:
            focal_loss *= self.alpha.gather(0, targets)

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        return focal_loss


class Asymmetric(nn.Module):
    def __init__(
        self,
        num_classes,
        gamma_neg=1.5,
        gamma_pos=1.0,
        clip=0.03,
        **params,
    ):
        """Asymmetric Loss for Multi-label Classification. https://arxiv.org/abs/2009.14119
        Loss function where negative classes are weighted less than the positive classes.
        Note: the inputs are logits and targets, not sigmoids.
        Usage:
            inputs = torch.randn(5, 3)
            targets = torch.randint(0, 1, (5, 3)) # must be binary
            loss_fn = AsymmetricLoss()
            loss = loss_fn(inputs, targets)
        Args:
            gamma_neg: loss attenuation factor for negative classes
            gamma_pos: loss attenuation factor for positive classes
            clip: shifts the negative class probability and zeros loss if probability > clip
            reduction: how to reduced final loss. Must be one of mean[default], sum, none
        """
        super().__init__()
        if clip < 0.0 or clip > 1.0:
            raise ValueError("Clipping value must be non-negative and less than one")
        if gamma_neg < gamma_pos:
            raise ValueError(
                "Need to ensure that loss for hard positive is penalised less than hard negative"
            )
        self.num_classes = num_classes
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def _get_binary_cross_entropy_loss_and_pt_with_logits(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
        pt = torch.exp(-ce_loss)  # probability at y_i=1
        return ce_loss, pt

    def forward(self, model, out, batch):
        preds = out["clique_logits"]
        targets = batch["clique"]
        if targets.size(-1) != self.num_classes:
            targets = F.one_hot(targets, self.num_classes).float()
        ce_loss, pt = self._get_binary_cross_entropy_loss_and_pt_with_logits(preds, targets)
        # shift and clamp (therefore zero gradient) high confidence negative cases
        pt_neg = (pt + self.clip).clamp(max=1.0)
        ce_loss_neg = -torch.log(pt_neg)
        loss_neg = (1 - pt_neg) ** self.gamma_neg * ce_loss_neg
        loss_pos = (1 - pt) ** self.gamma_pos * ce_loss
        loss = targets * loss_pos + (1 - targets) * loss_neg
        return loss.mean()


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size, margin, scale):
        """
        ArcFace: Additive Angular Margin Loss for Deep Face Recognition
        (https://arxiv.org/pdf/1801.07698.pdf)
        Args:
            num_classes: The number of classes in your training dataset
            embedding_size: The size of the embeddings that you pass into
            margin: m in the paper, the angular margin penalty in radians
            scale: s in the paper, feature scale
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale

        self.W = torch.nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_normal_(self.W)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (None, embedding_size)
            labels: (None,)
        Returns:
            loss: scalar
        """
        cosine = self.get_cosine(embeddings)  # (None, n_classes)
        mask = self.get_target_mask(labels)  # (None, n_classes)
        cosine_of_target_classes = cosine[mask == 1]  # (None, )
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes
        )  # (None, )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(
            1
        )  # (None,1)
        logits = cosine + (mask * diff)  # (None, n_classes)
        logits = self.scale_logits(logits)  # (None, n_classes)
        return nn.CrossEntropyLoss()(logits, labels)

    def get_cosine(self, embeddings):
        """
        Args:
            embeddings: (None, embedding_size)
        Returns:
            cosine: (None, n_classes)
        """
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.W))
        return cosine

    def get_target_mask(self, labels):
        """
        Args:
            labels: (None,)
        Returns:
            mask: (None, n_classes)
        """
        batch_size = labels.size(0)
        onehot = torch.zeros(batch_size, self.num_classes, device=labels.device)
        onehot.scatter_(1, labels.unsqueeze(-1), 1)
        return onehot

    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        """
        Args:
            cosine_of_target_classes: (None,)
        Returns:
            modified_cosine_of_target_classes: (None,)
        """
        eps = 1e-6
        # theta in the paper
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1 + eps, 1 - eps))
        return torch.cos(angles + self.margin)

    def scale_logits(self, logits):
        """
        Args:
            logits: (None, n_classes)
        Returns:
            scaled_logits: (None, n_classes)
        """
        return logits * self.scale


class CosineEmbedding(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.fn = nn.CosineEmbeddingLoss()

    def forward(self, model, out, batch):
        if (batch["clique"].ndimension() == 1) and (
            out["embedding"].size(-1) == out["pos_embedding"].size(-1) != 0
        ):
            labels = torch.ones(out["embedding"].size(0)).to(out["embedding"].device)
            return self.fn(out["embedding"], out["pos_embedding"], labels)
        return torch.tensor(0.0, requires_grad=True).to(out["embedding"].device)
