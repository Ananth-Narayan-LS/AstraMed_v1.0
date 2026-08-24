

import torch

def train_one_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
):
    """
        Train the model for one epoch.

        Parameters
        ----------
        model : nn.Module

        dataloader : DataLoader

        criterion : Loss Function

        optimizer : Optimizer

        device : torch.device

        Returns
        -------
        float
            Average training loss.
    """

    model.train()

    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)

        targets = (
            batch["target"]
            .float()
            .unsqueeze(1)
            .to(device)
        )

        # FORWARD
        outputs = model(images)
        loss = criterion(outputs, targets)

        # BACKPROPAGATION
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # STATS
        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss
