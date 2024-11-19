import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current macOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    print("MPS is available!")
    
    # Set the device
    mps_device = torch.device("mps")

    # Test Tensor creation and operations on MPS
    x = torch.ones(5, device=mps_device)
    print("Tensor on MPS:", x)

    y = x * 2
    print("Tensor after multiplication:", y)

    # Define a simple model
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc = nn.Linear(5, 3)

        def forward(self, x):
            x = self.fc(x)
            return F.relu(x)

    # Create the model and move it to the MPS device
    model = SimpleNet().to(mps_device)
    print("Model moved to MPS.")

    # Perform a forward pass with the model
    pred = model(x)
    print("Prediction from the model:", pred)

