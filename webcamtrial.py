import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)  # LOAD RESNET18
model.fc = nn.Linear(model.fc.in_features, 7)
model.eval()

transform = transforms.Compose([transforms.ToTensor()])

cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)

if cap.isOpened():  # try to get the first frame
    retention, frame = cap.read()
else:
    retention = False
    frame = "Not supporting video "

while retention:
    cv2.imshow("preview", frame)
    frame_pil = Image.fromarray(np.uint8(frame)).convert('RGB')
    frame_tensor = transform(frame_pil).unsqueeze(0)
    retention, frame = cap.read()
    x = model(frame_tensor)
    key = cv2.waitKey(100)
    print(x, x.argmax(dim=1))
    if key == 27:  # IF ESC KEY PRESSED
        break  # END RETENTION

cap.release()
cv2.destroyWindow("preview")
