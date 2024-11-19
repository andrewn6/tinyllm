import torch
import torch.nn as nn
import torch.nn.functional as F

def test_cpu():
    print("=== Testing CPU ===")
    cpu_device = torch.device("cpu")
    print("CPU device is available.")

    try:
        x = torch.ones(5, device=cpu_device)
        print("Tensor on CPU:", x)

        y = x * 2
        print("Tensor after multiplication:", y)
    except Exception as e:
        print("Error during tensor operations on CPU:", e)
        return

    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc = nn.Linear(5, 3)

        def forward(self, x):
            x = self.fc(x)
            return F.relu(x)

    try:
        model = SimpleNet().to(cpu_device)
        print("Model is on CPU.")
        pred = model(x)
        print("Prediction from the model:", pred)
    except Exception as e:
        print("Error during model operations on CPU:", e)

if __name__ == "__main__":
    test_cpu()

