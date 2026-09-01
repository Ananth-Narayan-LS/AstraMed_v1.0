

import torch

def validate(
        model,
        dataloader,
        criterion,
        device
):
    """
        Evaluate the model on a validation dataset.

        Parameters
        ----------
        model : torch.nn.Module
            Trained model.

        dataloader : DataLoader
            Validation DataLoader.

        criterion : loss function
            Loss function used for evaluation.

        device : torch.device
            CPU or GPU.

        Returns
        -------
        float
            Average validation loss.
    """
    # Evaluation mode
    model.eval()

    total_loss = 0.0
    total_samples = 0

    # No gradients during validation
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["target"].float().to(device)

            # Forward pass
            outputs = model(images)

            # Match target shape with model output
            targets = targets.view_as(outputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Accumulate weighted loss
            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size


    # Average loss over all validation samples
    average_loss = total_loss / total_samples

    return average_loss