from keras_facenet import FaceNet
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

embedder = FaceNet()

image = Image.open('./lab-data/me-data/me-sad.jpg').convert('RGB')
image_np = np.array(image)

detections = embedder.extract(image_np, threshold=0.95)

draw = ImageDraw.Draw(image)
for d in detections:
    box = d['box']
    x, y, width, height = box
    draw.rectangle([(x, y), (x+width, y+height)], outline='red', width=10)

plt.imshow(image)
plt.show()
