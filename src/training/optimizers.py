

import torch.optim as optim

def build_optimizer(
        model,
        optimizer_name : str = "adam",
        learning_rate : float = 1e-3
):
    """
        Build an optimizer.

        Parameters
        ----------
        model : nn.Module

        optimizer_name : str

        learning_rate : float

        Returns
        -------
        torch.optim.Optimizer
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr = learning_rate
        )

    elif optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr = learning_rate,
            momentum=0.9
        )

    else:
        raise ValueError(f"Unsupported optimizer : {optimizer_name}")
