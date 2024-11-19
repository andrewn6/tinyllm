import torch
import torch.nn as nn
import torch.nn.functional as F

def test_cuda():
    print("=== Testing NVIDIA CUDA ===")
    if not torch.cuda.is_available():
        print("CUDA is not available. Please ensure that you have an NVIDIA GPU with CUDA support and that the CUDA toolkit is installed.")
    else:
        print("CUDA is available!")
        cuda_device = torch.device("cuda")

        try:
            x = torch.ones(5, device=cuda_device)
            print("Tensor on CUDA:", x)

            y = x * 2
            print("Tensor after multiplication:", y)
        except Exception as e:
            print("Error during tensor operations on CUDA:", e)
            return

        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc = nn.Linear(5, 3)

            def forward(self, x):
                x = self.fc(x)
                return F.relu(x)

        try:
            model = SimpleNet().to(cuda_device)
            print("Model moved to CUDA.")

            pred = model(x)
            print("Prediction from the model:", pred)
        except Exception as e:
            print("Error during model operations on CUDA:", e)

if __name__ == "__main__":
    test_cuda()

