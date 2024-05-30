import pickle
import torch
import numpy
import time

with open("hw1.pkl", 'rb') as f:  # Open data (hw1.pkl) in binary read mode as file f
    data = pickle.load(f)  # Defining dataset via pickle.load
    print("\n")

x1_t = torch.from_numpy(data["train"]["x1"])  # Convert raw training data (sq.ftg.) to a tensor
x2_t = torch.from_numpy(data["train"]["x2"])  # Convert...data (yr.blt.) to...tensor
y_t = torch.from_numpy(data["train"]["y"]).float()  # Convert...data (price) to... tensor and float

X = torch.concat([x1_t.view(-1, 1), x2_t.view(-1, 1), torch.ones(500, 1)], dim=1)  # Concatenate (x1, x2, 1)
# (-1, 1) inferences the size based on the data, and concatenates to a single column
print(f"X: {X}")

XTX = X.T @ X  # Equation for Weights (XTX)
XTX1 = torch.linalg.inv(XTX)  # Equation for Weights (XTX)^-1

Wt = torch.zeros(3, requires_grad=True)

dif = torch.inf  # Placeholder value
current_loss = torch.inf  # Placeholder value
i_loss = torch.mean(torch.square((X @ Wt) - y_t))
print(f"i_loss: {i_loss}")

W = (XTX1 @ X.T) @ y_t
print(torch.mean(torch.square((X @ W) - y_t)))

while dif > 1:
    Wt = Wt - (1 / 500) * (1e-7) * ((XTX @ Wt) - (X.T @ y_t))
    current_loss = torch.mean(torch.square((X @ Wt) - y_t))
    print(f"current_loss: {current_loss}")
    dif = i_loss - current_loss
    print(f"dif: {dif}")
    i_loss = current_loss



# x1_tt = torch.from_numpy(data["test"]["x1"])
# x2_tt = torch.from_numpy(data["test"]["x2"])
# y_tt = torch.from_numpy(data["test"]["ytest"]).float()

# Xt = torch.concat([x1_tt.view(-1, 1), x2_tt.view(-1, 1), torch.ones(y_tt.size(0), 1)], dim=1)
# XWY = torch.square((Xt @ W) - y_tt)
# MSE = torch.mean(XWY)
# print(MSE)
# print(torch.sqrt(MSE))

# new_data = torch.tensor([1941, 2005, 1]).float()
# test2 = new_data @ W
# print(test2)
