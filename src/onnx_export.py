import torch
import torchvision.models as models
import torch.nn as nn

NUM_CLASSES = 101

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load('../model/model.pth', map_location='cpu'))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model, dummy_input, '../model/model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=18
)
print("Export done.")
