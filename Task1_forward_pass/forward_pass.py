
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv( "D:\Semester4_1\Deep_Learning\Task1_forward_pass\data_banknote_authentication.csv")

# Features & Target
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

num_features = X.shape[1]

w = np.random.randn(num_features)

lr = 0.0001
epochs = 1000

print("Initial Weights:", w)


for epoch in range(epochs):

    total_loss = 0

    for i in range(len(X)):

        x = X[i]
        target = y[i]

        prediction = np.dot(w, x)
        loss = (prediction - target) ** 2
        total_loss += loss
        grad_w = 2 * (prediction - target) * x
        w = w - lr * grad_w

    if epoch % 100 == 0:

        avg_loss = total_loss / len(X)

        print(
            f"Epoch {epoch} | Loss = {avg_loss:.4f}"
        )

print("\nTraining Finished")
print("Final Weights:", w)

correct = 0

for i in range(len(X)):
    pred = np.dot(w, X[i])

    if pred >= 0.5:
        predicted_class = 1
    else:
        predicted_class = 0

    if predicted_class == y[i]:
        correct += 1

accuracy = (correct / len(X)) * 100

print(f"\nAccuracy = {accuracy:.2f}%")