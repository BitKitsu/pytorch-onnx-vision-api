import torch
import torchvision.models as models
import torch.nn as nn
import onnxruntime as ort
import numpy as np

NUM_CLASSES = 101

# PyTorch
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load('../model/model.pth', map_location='cpu'))
model.eval()

dummy = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    pt_out = model(dummy).numpy()

# ONNX
sess = ort.InferenceSession('../model/model.onnx',
       providers=['CPUExecutionProvider'])
ort_out = sess.run(None, {'input': dummy.numpy()})[0]

# Compare
pt_class = int(np.argmax(pt_out))
ort_class = int(np.argmax(ort_out))
max_diff = float(np.max(np.abs(pt_out - ort_out)))

print(f"PyTorch class: {pt_class}")
print(f"ONNX Runtime class: {ort_class}")
print(f"Max diff: {max_diff:.6f}")
print(f"Compatibility: {'OK' if pt_class == ort_class else 'INCOMPATIBILE'}")
