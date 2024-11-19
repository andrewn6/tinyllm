import torch
import torch.nn as nn
import torch.nn.functional as F

def test_mps():
    print("=== Testing MPS (Metal Performance Shaders) ===")
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not built with MPS enabled.")
        else:
            print("MPS not available because the current macOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
    else:
        print("MPS is available!")
        mps_device = torch.device("mps")

        try:
            x = torch.ones(5, device=mps_device)
            print("Tensor on MPS:", x)

            y = x * 2
            print("Tensor after multiplication:", y)
        except Exception as e:
            print("Error during tensor operations on MPS:", e)
            return

        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc = nn.Linear(5, 3)

            def forward(self, x):
                x = self.fc(x)
                return F.relu(x)

        try:
            model = SimpleNet().to(mps_device)
            print("Model moved to MPS.")

            pred = model(x)
            print("Prediction from the model:", pred)
        except Exception as e:
            print("Error during model operations on MPS:", e)

if __name__ == "__main__":
    test_mps()

