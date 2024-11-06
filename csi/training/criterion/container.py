from torch import nn

from csi.training.criterion.weight_strategy import WeightStrategy


class CriterionContainer(nn.Module):
    def __init__(
        self, criterions: list[nn.Module], weight_strategies: list[WeightStrategy], device=None
    ) -> None:
        super().__init__()
        self.criterions = [c.to(device) for c in criterions] if device is not None else criterions
        self.weight_strategies = weight_strategies

    def forward(self, model, out, target, iteration):
        criterion_losses = {
            criterion.__class__.__name__: weight_strategy(iteration)
            * criterion(model, out, target)
            for criterion, weight_strategy in zip(self.criterions, self.weight_strategies)
        }
        loss_optimizers = [
            criterion.loss_optimizer
            for criterion in self.criterions
            if hasattr(criterion, "loss_optimizer") and criterion.loss_optimizer is not None
        ]
        return {
            "criterion_losses": criterion_losses,
            "total_loss": sum(criterion_losses.values()),
            "loss_optimizers": loss_optimizers,
        }

    def __iter__(self):
        return iter(self.criterions)
