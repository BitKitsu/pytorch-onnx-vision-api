# pytorch-onnx-vision-api

> Transfer learning image classifier (ResNet18) trained on [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101), exported to ONNX and served via FastAPI in Docker. Includes PyTorch vs ONNX Runtime CPU inference benchmark.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
  - [1. Train the model](#1-train-the-model)
  - [2. Export to ONNX](#2-export-to-onnx)
  - [3. Verify predictions](#3-verify-predictions)
  - [4. Benchmark](#4-benchmark)
  - [5. Run with Docker](#5-run-with-docker)
- [API Reference](#api-reference)
- [How it works](#how-it-works)

---

## Overview

This project demonstrates a full ML deployment pipeline:

1. **Transfer learning** - ResNet18 (pretrained on ImageNet) fine-tuned on [Food-101](https://pytorch.org/vision/stable/generated/torchvision.datasets.Food101.html) (101 food categories, ~101k images)
2. **ONNX export** - model converted to ONNX format for portable, optimized inference
3. **Inference benchmark** - PyTorch CPU and ONNX Runtime CPU comparison
4. **REST API** - FastAPI endpoint accepting image uploads and returning predictions
5. **Docker deployment** - containerized, ready to deploy

---

## Results

### Training (ResNet18, 10 epochs)

|Epoch|Val Accuracy|
|---|---|
|1|70.4%|
|2|75.4%|
|3|75.9%|
|4|75.5%|
|5|76.6%|
|6|76.8%|
|7|75.9%|
|8|76.2%|
|9|75.1%|
|10|**76.0%**|

### Inference Benchmark (CPU)

|Runtime|Time per image|Speedup|
|---|---|---|
|PyTorch CPU|184.25 ms|1x|
|ONNX Runtime CPU|4.37 ms|**42.15x**|

ONNX Runtime is **42x faster** than PyTorch on CPU inference - making it more practical for production deployment without a GPU.

---

## Project Structure

```

|-- README.md
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- src/
|   |-- train.py
|   |-- export_onnx.py
|   |-- benchmark.py
|   |-- app.py
|-- model/
|   |-- .gitkeep
|-- data/
|   |-- .gitkeep
|-- raport/
    |-- raport.md
    |-- raport.pdf
```

> **Note:** `model.onnx` is available as a [GitHub Release](../../releases) attachment.  
> `data/` is downloaded automatically on first run via torchvision.

---

## Requirements

- Python 3.11+
- CUDA-capable GPU (optional)
- Docker and Docker Compose

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quickstart

### 1. Train the model

Downloads Food-101 (~5 GB) automatically on first run.

```bash
python src/train.py
```

Output:
```
Using: cuda
Epoch 1 | Val Acc: 0.704
...
Epoch 10 | Val Acc: 0.760
Model saved.
```

Saves `model.pth` to the `model/` directory.

---

### 2. Export to ONNX

```bash
python src/export_onnx.py
```

Saves `model.onnx` to the `model/` directory.

---

### 3. Verify predictions
 
Checks that PyTorch and ONNX Runtime return identical results on the same input.
 
```bash
python src/verify.py
```
 
Output:
```
PyTorch class: 26
ONNX Runtime class: 26
Max diff: 0.000005
Compatibility: OK
```
 
The maximum difference between raw logits is `0.000005` - within floating-point numerical error.
 
---


### 4. Benchmark

Compares PyTorch CPU and ONNX Runtime CPU inference time over 100 runs.

```bash
python src/benchmark.py
```

Output:
```
PyTorch CPU: 184.25 ms/image
ONNX Runtime CPU: 4.37 ms/image
Speedup: 42.15x
```

---

### 5. Run with Docker

Make sure `model/model.onnx` exists

```bash
docker compose up -d
```

The API will be available at `http://localhost:3333`.

To stop:

```bash
docker-compose down
```

---

## API Reference

### `GET /health`

Check if the service is running

```bash
curl http://localhost:3333/health
```

```json
{ "status": "ok" }
```

---

### `POST /predict`

Upload an image and get a predicted Food-101 class

```bash
curl -X POST http://localhost:8000/predict -F "file=@pizza.jpg"
```

```json
{
  "class_id": 72,
  "confidence": 0.9421
}
```

|Field|Type|Description|
|---|---|---|
|`class_id`|int|Predicted class index|
|`confidence`|float|Softmax confidence score (0.0–1.0)|

---

ResNet18 was originally pretrained on ImageNet (1000 general categories). The final fully-connected layer was replaced and retrained on Food-101 - a dataset of 101 food categories not present as distinct classes in ImageNet.
