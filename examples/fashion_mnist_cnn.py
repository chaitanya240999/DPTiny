import random
import time

import numpy as np

from dptiny import (
    Variable,
    is_available,
    is_gpu,
    no_grad,
    softmax_cross_entropy,
    test_mode,
    to_cpu,
    to_gpu,
    use_gpu,
    xp,
)
from dptiny.data import DataLoader
from dptiny.data.fashion_mnist import CLASSES, get_fashion_mnist
from dptiny.nn import (
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from dptiny.optim import Adam

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

if is_available():
    use_gpu()
    xp.random.seed(SEED)
    print("GPU enabled for training.")
else:
    print("GPU not available; training on CPU.")

print("Loading Fashion-MNIST dataset...")
X_train, X_test, y_train, y_test = get_fashion_mnist(flatten=False)

# Standardize using statistics computed only from the training images.
train_mean = float(X_train.mean())
train_std = float(X_train.std())
X_train = ((X_train - train_mean) / train_std).astype(xp.float32)
X_test = ((X_test - train_mean) / train_std).astype(xp.float32)

if is_gpu():
    X_train = to_gpu(X_train)
    X_test = to_gpu(X_test)
    y_train = to_gpu(y_train)
    y_test = to_gpu(y_test)

# Baseline architecture from examples/mnist_cnn.py, unchanged.
model = Sequential(
    Conv2d(1, 16, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Conv2d(16, 32, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(32 * 7 * 7, 128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 10),
)
if is_gpu():
    model.to_gpu()

batch_size = 64
max_epoch = 10

data_loader = DataLoader((X_train, y_train), batch_size)
train_eval_loader = DataLoader(
    (X_train, y_train), batch_size, shuffle=False
)
test_loader = DataLoader(
    (X_test, y_test), batch_size, shuffle=False
)

optimizer = Adam(model, lr=0.001)


def evaluate(loader):
    correct = 0
    total = 0
    with test_mode(), no_grad():
        for x, t in loader:
            y = model(Variable(x))
            pred = y.data.argmax(axis=1)
            correct += int(to_cpu((pred == t).sum()))
            total += len(t)
    return correct / total


start_time = time.time()

for epoch in range(max_epoch):
    sum_loss = 0.0
    train_count = 0
    model.train()

    for x, t in data_loader:
        x = Variable(x)
        y = model(x)
        loss = softmax_cross_entropy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(to_cpu(loss.data)) * len(t)
        train_count += len(t)

    train_loss = sum_loss / train_count
    train_acc = evaluate(train_eval_loader)
    test_acc = evaluate(test_loader)

    print(
        f"Epoch {epoch + 1:2d}/{max_epoch} | "
        f"train loss: {train_loss:.4f} | "
        f"train acc: {train_acc:.4f} | "
        f"test acc: {test_acc:.4f}"
    )

# Final predictions for the confusion matrix.
confusion = np.zeros((10, 10), dtype=np.int64)

with test_mode(), no_grad():
    for x, t in test_loader:
        y = model(Variable(x))
        pred = to_cpu(y.data.argmax(axis=1))
        true = to_cpu(t)
        np.add.at(confusion, (true, pred), 1)

print("\nConfusion matrix (rows=true, columns=predicted):")
print(confusion)

print("\nPer-class accuracy:")
class_accuracy = np.diag(confusion) / confusion.sum(axis=1)
for i, class_name in enumerate(CLASSES):
    print(f"{i}: {class_name:12s} {class_accuracy[i]:.4f}")

# Treat confusion as an unordered class pair: A->B and B->A are combined.
pair_confusion = confusion + confusion.T
np.fill_diagonal(pair_confusion, 0)
i, j = np.unravel_index(np.argmax(pair_confusion), pair_confusion.shape)

print(
    "\nMost confused pair: "
    f"{CLASSES[i]} <-> {CLASSES[j]} "
    f"({pair_confusion[i, j]} total confusions: "
    f"{CLASSES[i]}->{CLASSES[j]}={confusion[i, j]}, "
    f"{CLASSES[j]}->{CLASSES[i]}={confusion[j, i]})"
)

print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
