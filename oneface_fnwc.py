from keras_facenet import FaceNet
from PIL import Image, ImageDraw
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, CenterCrop
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # DO NOT PRINT USER WARNINGS

embedder = FaceNet()  # IMPORT FACENET EMBEDDER

cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)
retention, frame = cap.read()

stdout_ref = sys.__stdout__
null_f = open('/dev/null', 'w')

transform = Compose([Resize(256), CenterCrop(224), ToTensor(), Normalize(0.5077, 0.2550)])

emotion = ["\033[31mangry\033[0m", "\033[32mdisgusted\033[0m", "\033[35mfearful\033[0m", "\033[33mhappy\033[0m",
           "\033[37mneutral\033[0m", "\033[34msad\033[0m", "\033[36msurprised\033[0m"]

emotion_trump = None
past_emotions = []

with torch.no_grad():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)  # LOADING RESNET
    model.fc = nn.Linear(model.fc.in_features, 7)  # DEFINING CLASS NUMBER

    state_dict = torch.load('resnet18_model_file_mixup_on.pth', map_location='cpu')  # LOADING WEIGHTS
    model.load_state_dict(state_dict)  # LOADING THE MODEL WITH SET WEIGHTS
    model.eval()  # REMOVE DROPOUT

    while retention:  # WHILE CAPTURING
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # CONVERTS IMAGE TO PIL
        frame_tensor = transform(frame_pil).unsqueeze(0)

        sys.stdout = null_f
        detections = embedder.extract(frame, threshold=0.95)
        sys.stdout = stdout_ref

        draw = ImageDraw.Draw(frame_pil)
        for d in detections:
            box = d['box']
            x, y, width, height = box
            draw.rectangle([(x, y), (x + width, y + height)], outline='red', width=10)
            region = frame_pil.crop((x, y, x+width, y+height))
            region_tensor = transform(region).unsqueeze(0)
            z = model(region_tensor)
            emotion_index = torch.argmax(z, dim=1).item()
            choice = emotion[emotion_index]
            past_emotions.append(choice)
            if len(past_emotions) > 3:
                past_emotions.pop(0)
            if len(set(past_emotions)) == 1:
                emotion_trump = past_emotions[0]

        sys.stdout.write("\rCurrent Emotion: {}".format(emotion_trump))
        sys.stdout.flush()

        frame_annotated = np.array(frame_pil)

        cv2.imshow("preview", frame_annotated)  # DISPLAY EACH FRAME

        retention, frame = cap.read()  # SAVE FRAMES
        key = cv2.waitKey(100)  # FRAME RATE
        if key == 27:  # IF ESC KEY PRESSED
            break  # END RETENTION

    cap.release()  # RELEASE CAMERA
    cv2.destroyWindow("preview")  # CLOSE OUTSIDE PREVIEW