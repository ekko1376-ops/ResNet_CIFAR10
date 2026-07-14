import os
import re
import matplotlib.pyplot as plt

log_path = "train.txt"

epochs = []
train_losses = []
train_accs = []
test_losses = []
test_accs = []

current_epoch = None

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        epoch_match = re.search(r"Epoch\[(\d+)/\d+\]", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        train_loss_match = re.search(r"Train Loss: ([\d.]+)", line)
        if train_loss_match:
            epochs.append(current_epoch)
            train_losses.append(float(train_loss_match.group(1)))

        train_acc_match = re.search(r"Train Acc: ([\d.]+)%", line)
        if train_acc_match:
            train_accs.append(float(train_acc_match.group(1)))

        test_loss_match = re.search(r"Test Loss: ([\d.]+)", line)
        if test_loss_match:
            test_losses.append(float(test_loss_match.group(1)))

        test_acc_match = re.search(r"Test Acc: ([\d.]+)%", line)
        if test_acc_match:
            test_accs.append(float(test_acc_match.group(1)))

os.makedirs("figures", exist_ok=True)

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("figures/loss_curve.png", dpi=300)

plt.figure()
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, test_accs, label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.savefig("figures/accuracy_curve.png", dpi=300)

print("Saved figures to figures/")