import math

from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR

def create_optimizer(
    parameters,
    learning_rate: float = 1.0,
) -> Optimizer:
    """
    Create the Adam optimizer.
    """
    return Adam(
        parameters,
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

def create_noam_scheduler(
    optimizer: Optimizer,
    d_model: int,
    warmup_steps: int = 4000,
    scale: float = 1.0,
) -> LambdaLR:
    """
    Create the Transformer Noam learning-rate scheduler.
    """

    def lr_lambda(step):
        step = max(step, 1)

        return scale * (
            d_model**-0.5
            * min(
                step**-0.5,
                step * warmup_steps**-1.5,
            )
        )

    return LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )