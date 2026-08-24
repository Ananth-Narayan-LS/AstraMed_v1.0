

#Loss functions used for training AstraMed models.


import torch.nn as nn

def build_loss(loss_name: str = "bce"):
    """
        Build a loss function.

        Parameters
        ----------
        loss_name : str
            Name of the loss function.

        Returns
        -------
        nn.Module
            PyTorch loss function.
            
    """
    loss_name = loss_name.lower()

    if loss_name == "bce":
        return nn.BCELoss()
 
    # Currently our model ends with
    # nn.Sigmoid()
    # Therefore
    # nn.BCELoss()
    # is correct.
    # Later, when we remove the sigmoid layer (which is recommended for better numerical stability), we will simply switch to:
    # nn.BCEWithLogitsLoss()
    # without changing the training code.


    elif loss_name == "bce_logits":
        return nn.BCEWithLogitsLoss()

    else:
        raise ValueError(f"Unsupported loss function : {loss_name}")
     

