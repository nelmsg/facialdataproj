import pickle
import torch
import numpy
import time
import matplotlib.pyplot as plt

with open("hw1.pkl", 'rb') as f:  # Open data (hw1.pkl) in binary read mode as file f
    data = pickle.load(f)  # Defining dataset via pickle.load
    print("\n")

x1_t = torch.from_numpy(data["train"]["x1"])  # Convert raw training data (sq.ftg.) to a tensor
x2_t = torch.from_numpy(data["train"]["x2"])  # Convert...data (yr.blt.) to...tensor
y_t = torch.from_numpy(data["train"]["y"]).float()  # Convert...data (price) to... tensor and float

X = torch.concat([x1_t.view(-1, 1), x2_t.view(-1, 1)], dim=1).float()  # Concatenate (x1, x2,)
# (-1, 1) inferences the size based on the data, and concatenates to a single column
print(f"X: {X}")

Wt = torch.zeros(3, requires_grad=True)
model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0)

dif = torch.inf  # Placeholder value
current_loss = torch.inf  # Placeholder value
i_loss = torch.inf
print(f"i_loss: {i_loss}")

losses = []

while dif > 1e7:
    optimizer.zero_grad()
    predictions = model(X)
    current_loss = torch.nn.functional.mse_loss(predictions, y_t.view(-1, 1))
    print(f"current_loss: {current_loss}")
    current_loss.backward()
    optimizer.step()
    dif = i_loss - current_loss
    print(f"dif: {dif}")
    i_loss = current_loss
    losses.append(current_loss.item())

plt.plot(range(len(losses)), losses)
plt.ylim(top=0.5e10)
plt.show()
