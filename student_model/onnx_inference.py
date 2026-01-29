import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import time

# Load ONNX model
session = ort.InferenceSession('/kaggle/working/tinydefectnet_student/checkpoints/student_model.onnx')

class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/test/Crazing/Cr_1.bmp').convert('RGB')
image_tensor = transform(image).unsqueeze(0).numpy()

# Time the inference
start_time = time.perf_counter()
outputs = session.run(None, {'input': image_tensor})
end_time = time.perf_counter()

inference_time_ms = (end_time - start_time) * 1000

logits = outputs[0][0]

# Get prediction
probs = np.exp(logits) / np.sum(np.exp(logits))
pred_idx = np.argmax(probs)

print(f"Prediction: {class_names[pred_idx]}")
print(f"Confidence: {probs[pred_idx]:.2%}")
print(f"Inference time: {inference_time_ms:.2f} ms")