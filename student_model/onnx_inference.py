import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import time

# Load ONNX model
session = ort.InferenceSession('/kaggle/working/tinydefectnet_student/checkpoints/student_model.onnx')

class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess once
image = Image.open('/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/test/Crazing/Cr_1.bmp').convert('RGB')
image_tensor = transform(image).unsqueeze(0).numpy()

# Benchmark parameters
NUM_WARMUP = 10  # Warmup runs
NUM_RUNS = 100   # Benchmark runs

print("Warming up...")
for _ in range(NUM_WARMUP):
    _ = session.run(None, {'input': image_tensor})

print(f"Benchmarking ({NUM_RUNS} runs)...")
times = []
for _ in range(NUM_RUNS):
    start = time.perf_counter()
    outputs = session.run(None, {'input': image_tensor})
    end = time.perf_counter()
    times.append((end - start) * 1000)

times = np.array(times)

# Get final prediction
logits = outputs[0][0]
probs = np.exp(logits) / np.sum(np.exp(logits))
pred_idx = np.argmax(probs)

print("\n" + "="*60)
print("BENCHMARK RESULTS")
print("="*60)
print(f"Prediction: {class_names[pred_idx]}")
print(f"Confidence: {probs[pred_idx]:.2%}")
print("\n" + "-"*60)
print("Inference Time Statistics:")
print("-"*60)
print(f"Mean:      {np.mean(times):6.2f} ms")
print(f"Median:    {np.median(times):6.2f} ms")
print(f"Std Dev:   {np.std(times):6.2f} ms")
print(f"Min:       {np.min(times):6.2f} ms")
print(f"Max:       {np.max(times):6.2f} ms")
print(f"P95:       {np.percentile(times, 95):6.2f} ms")
print(f"P99:       {np.percentile(times, 99):6.2f} ms")
print("="*60)

# Check if within target
TARGET_LATENCY = 50.0  # ms
if np.mean(times) <= TARGET_LATENCY:
    print(f"✓ Within target latency ({TARGET_LATENCY} ms)")
else:
    print(f"✗ Exceeds target latency ({TARGET_LATENCY} ms)")