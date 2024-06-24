from keras_facenet import FaceNet
from PIL import Image, ImageDraw
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, CenterCrop
import sys
import warnings
from dataclasses import dataclass
import os


@dataclass
class Face:
    location: int
    emotion: str
    distance: list


@dataclass
class FaceData:
    location: int
    emotions_past: list


def get_detections(embedder, frame, frame_pil):
    sys.stdout = null_f
    detections = embedder.extract(frame, threshold=0.95)
    sys.stdout = stdout_ref
    faces_per_frame = []

    for d in detections:
        box = d['box']
        x, y, width, height = box
        location = x
        draw.rectangle([(x, y), (x + width, y + height)], outline='red', width=10)

        region = frame_pil.crop((x, y, x+width, y+height))
        region_tensor = transform(region).unsqueeze(0)
        z = model(region_tensor)
        # print(f"z: {z}")

        emotion_num = torch.argmax(z, dim=1).item()
        # print(emotion_num)
        emotion = emotion_list[emotion_num]
        faces_per_frame.append(Face(location, emotion, []))
        # print(f"faces_per_frame: {faces_per_frame}")
    return faces_per_frame


def find_distance(face_data, faces_per_frame):
    # print(f"face data: {face_data}")
    for item in face_data:
        for detection in faces_per_frame:
            detection.distance.append(abs(item.location - detection.location))
    counter = 0
    for piece in face_data:
        counter += 1
        if len(set(piece.emotions_past)) == 1 and len(piece.emotions_past) == 3:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Face {counter}: {set(piece.emotions_past)}")


def match_faces(face_data, faces_per_frame):
    if len(face_data) == 0:
        faces_per_frame_sorted = faces_per_frame
    else:
        faces_per_frame_sorted = sorted(faces_per_frame, key=lambda item: min(item.distance))
    associations = {}
    for f in faces_per_frame_sorted:
        sorted_distances = sorted(list(enumerate(f.distance)), key=lambda item: item[1])
        is_associated = False
        for d in sorted_distances:
            if d[0] not in associations:
                is_associated = True
                associations[d[0]] = f
                face_data[d[0]].emotions_past.append(f.emotion)
                face_data[d[0]].location = f.location
                break
        if not is_associated:
            face_data.append(FaceData(f.location, [f.emotion]))
            associations[len(face_data) - 1] = f
    remove_indices = []
    for item, f in enumerate(face_data):
        if item not in associations:
            remove_indices.append(item)
    remove_indices_sorted = sorted(remove_indices, key=lambda item: item, reverse=True)
    for i in remove_indices_sorted:
        face_data.pop(i)


os.environ['TERM'] = 'xterm'

warnings.filterwarnings("ignore", category=UserWarning)  # DO NOT PRINT USER WARNINGS
stdout_ref = sys.__stdout__
null_f = open('/dev/null', 'w')

embedder = FaceNet()  # IMPORT FACENET EMBEDDER

cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)
retention, frame = cap.read()

transform = Compose([Resize(256), CenterCrop(224), ToTensor(), Normalize(0.5077, 0.2550)])

emotion_list = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

face_data = []

distance = 0

with torch.no_grad():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)  # LOADING RESNET
    model.fc = nn.Linear(model.fc.in_features, 7)  # DEFINING CLASS NUMBER

    state_dict = torch.load('resnet18_model_file_mixup_on_orig.pth', map_location='cpu')  # LOADING WEIGHTS
    model.load_state_dict(state_dict)  # LOADING THE MODEL WITH SET WEIGHTS
    model.eval()  # REMOVE DROPOUT

    while retention:  # WHILE CAPTURING
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # CONVERTS IMAGE TO PIL
        # frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE))  # CONVERTS IMAGE TO PIL
        frame_tensor = transform(frame_pil).unsqueeze(0)

        draw = ImageDraw.Draw(frame_pil)

        faces_per_frame = get_detections(embedder, frame, frame_pil)
        find_distance(face_data, faces_per_frame)
        match_faces(face_data, faces_per_frame)

        for i in face_data:
            # print(f"emotions past: {i.emotions_past}")
            if len(i.emotions_past) > 3:
                i.emotions_past.pop(0)

        frame_annotated = np.array(frame_pil)
        frame_annotated_bgr = cv2.cvtColor(frame_annotated, cv2.COLOR_RGB2BGR)
        cv2.imshow("preview", frame_annotated_bgr)  # DISPLAY EACH FRAME

        retention, frame = cap.read()  # SAVE FRAMES
        key = cv2.waitKey(100)  # FRAME RATE
        if key == 27:  # IF ESC KEY PRESSED
            break  # END RETENTION

    cap.release()  # RELEASE CAMERA
    cv2.destroyWindow("preview")  # CLOSE OUTSIDE PREVIEW
