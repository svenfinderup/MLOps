import torch
import typer
from data import corrupt_mnist
from sven_project_day_2.model import MyAwesomeModel
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = typer.Typer()

@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10):
    """Train the model."""
    print("Training started...")  # Added print to indicate the start of training
    train_set, _ = corrupt_mnist()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    model = MyAwesomeModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (predictions.argmax(1) == labels).sum().item()

            # Print training loss and accuracy every 100 steps (adjust based on your data size)
            if step % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # At the end of each epoch, log average loss and accuracy
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {total_correct/len(train_set):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to 'model.pth'")

@app.command()
def evaluate(model_checkpoint: str):
    """Evaluate the model."""
    print(f"Evaluating model: {model_checkpoint}...")  # Added to indicate evaluation started
    _, test_set = corrupt_mnist()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    model.eval()
    total_correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            predictions = model(images)
            total_correct += (predictions.argmax(1) == labels).sum().item()
            total += labels.size(0)

    # Print test accuracy
    accuracy = total_correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    app()
