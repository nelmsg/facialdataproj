import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, CenterCrop
import cv2
from PIL import Image

transform = Compose([Resize(256), CenterCrop(224), ToTensor(), Normalize(0.5077, 0.2550)])

cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)

if cap.isOpened():
    retention, frame = cap.read()
else:
    retention = False
    frame = "Not supporting video "

with torch.no_grad():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)  # LOADING RESNET
    model.fc = nn.Linear(model.fc.in_features, 7)  # DEFINING CLASS NUMBER

    state_dict = torch.load('resnet18_model_file_mixup_on.pth', map_location='cpu')  # LOADING WEIGHTS
    model.load_state_dict(state_dict)  # LOADING THE MODEL WITH SET WEIGHTS
    model.eval()  # REMOVE DROPOUT

    while retention:  # WHILE CAPTURING
        cv2.imshow("preview", frame)  # DISPLAY EACH FRAME
        frame_pil = Image.fromarray(np.uint8(frame)).convert('L').convert('RGB')  # CONVERTS IMAGE TO PIL
        frame_tensor = transform(frame_pil).unsqueeze(0)  # CONVERTS PIL TO TENSOR
        retention, frame = cap.read()  # SAVE FRAMES
        z = model(frame_tensor)  # CLASS PROBABILITIES PER FRAME
        key = cv2.waitKey(100)  # FRAME RATE
        print(z.argmax(dim=1).item())  # PRINTING MOST LIKELY CLASS
        if key == 27:  # IF ESC KEY PRESSED
            break  # END RETENTION

    cap.release()  # RELEASE CAMERA
    cv2.destroyWindow("preview")  # CLOSE OUTSIDE PREVIEW
