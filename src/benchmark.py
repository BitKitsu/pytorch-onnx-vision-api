import torch, time, numpy as np
import torchvision.models as models
import torch.nn as nn
import onnxruntime as ort

NUM_CLASSES = 101
dummy = torch.randn(1, 3, 224, 224)

# PyTorch CPU
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load('../model/model.pth', map_location='cpu'))
model.eval()

N = 100
start = time.time()
with torch.no_grad():
    for _ in range(N):
        model(dummy)
pt_time = (time.time() - start) / N * 1000
print(f"PyTorch CPU: {pt_time:.2f} ms/image")

# ONNX Runtime CPU
sess = ort.InferenceSession('../model/model.onnx',
       providers=['CPUExecutionProvider'])
dummy_np = dummy.numpy()

start = time.time()
for _ in range(N):
    sess.run(None, {'input': dummy_np})
ort_time = (time.time() - start) / N * 1000
print(f"ONNX Runtime CPU: {ort_time:.2f} ms/image")
print(f"Acceleration: {pt_time/ort_time:.2f}x")
