import torch
import typer
from sven_project_day_2.data import corrupt_mnist
from sven_project_day_2.model import MyAwesomeModel

# Device configuration (using CUDA if available, otherwise using CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(f"Model checkpoint: {model_checkpoint}")

    # Initialize the model and load the trained weights
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    # Load the test set
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    # Evaluate the model
    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    
    # Print the test accuracy
    print(f"Test accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    typer.run(evaluate)
