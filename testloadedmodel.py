import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, CenterCrop
from PIL import Image
import numpy as np

transform = Compose([Resize(256), CenterCrop(224), ToTensor(), Normalize(0.5077, 0.2550)])

with torch.no_grad():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 7)

    state_dict = torch.load('resnet18_model_file_mixup_on.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    crop_type = ["", "-crop"]
    person = ["me", "ash", "chayne"]
    emotion = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    individual_correct = {p: 0 for p in person}

    for w in crop_type:
        individual_correct = {p: 0 for p in person}
        for i in person:
            for x in emotion:
                file = Image.open(f"./lab-data/{i}-data{w}/{i}-{x}{w}.jpg")
                file_pil = Image.fromarray(np.uint8(file)).convert('L').convert('RGB')
                file_tensor = transform(file_pil).unsqueeze(0)

                z = model(file_tensor)
                prediction_index = z.argmax(dim=1).item()
                prediction = emotion[prediction_index]

                actual_class = emotion.index(x)
                if prediction_index == actual_class:
                    individual_correct[i] += 1

                print(f"{i}-{x}{w}:   {prediction}")
            print("")
        print(f"Individual Correct Predictions:")
        for p, correct_count in individual_correct.items():
            print(f"  {p}{w}: {correct_count}")
        print("")
