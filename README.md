# Tomato Leaf Disease Detection

Detect tomato leaves in an image with **YOLOv8** and classify each detected leaf as **Bacterial Spot**, **Early Blight**, or **Healthy** with a **ResNet18** classifier. Weights are hosted on Hugging Face and auto-downloaded at runtime — no manual downloads needed.

**Hugging Face weights:** https://huggingface.co/GovRang/tomato-leaf-model

## 🚀 Quick Start

### 1) Clone the Repository
```bash
git clone https://github.com/GovindRangappa/Tomato-Leaf-Disease-Computer-Vision-Model-.git
cd Tomato-Leaf-Disease-Computer-Vision-Model-
```

### 2) (Optional) Create a Virtual Environment
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate

# macOS/Linux
source .venv/bin/activate
```

### 3) Install Dependencies
```bash
pip install -r requirements.txt
```

### 4) Run on the Sample Image
```bash
python src/DetectCropClassify.py
```

**Note:** The first run will download the YOLO + classifier weights from Hugging Face. Outputs (crops + annotated image) are saved to `detected_and_classified/`.

## 🧱 Project Structure

```
.
├── src/
│   └── DetectCropClassify.py      # Detect → Crop → Classify pipeline
├── examples/
│   └── DetectionTest.jpg          # Sample input image
├── requirements.txt
├── README.md
└── .gitignore
```

## 🖼️ Example Results

*(Add screenshots after you run once — these paths will exist on first run)*

* **Input:** `examples/DetectionTest.jpg`
* **Output (annotated):** `detected_and_classified/annotated_image.jpg`
* **Crops:** `detected_and_classified/crop_*.jpg`

```
examples/DetectionTest.jpg → detected_and_classified/annotated_image.jpg
```

## 🧠 Models

* **Detector:** YOLOv8 (Ultralytics), weights: `best.pt`
* **Classifier:** ResNet18 (torchvision), 3 classes: `["Bacterial_Spot", "Early_Blight", "Healthy"]`, weights: `merged_model.pth`
* **Weights host:** `GovRang/tomato-leaf-model`

The script uses:

```python
from huggingface_hub import hf_hub_download

YOLO_MODEL_PATH = hf_hub_download("GovRang/tomato-leaf-model", "best.pt")
CLASSIFIER_MODEL_PATH = hf_hub_download("GovRang/tomato-leaf-model", "merged_model.pth")
```

## ⚙️ Inference Details

* **Preprocessing:** `Resize(256,256)` → `ToTensor()` → `Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])`
* **Prediction:** Softmax over 3 classes, label shown as `Class (confidence)`
* **Output:** Annotated bounding boxes + per-leaf crops

## 📊 Performance

Add your evaluation metrics here once finalized:

* **Detection:** mAP@0.5 = `TODO` on `YOUR_DATASET`
* **Classification:** Accuracy = `TODO` on `YOUR_TEST_SPLIT`

*(Tip: include dataset description and how you split/train.)*

## 🛠️ Troubleshooting

* **`ModuleNotFoundError`** → Ensure you installed dependencies: `pip install -r requirements.txt`
* **Slow first run** → Weights download from Hugging Face; subsequent runs use the local cache
* **CPU vs GPU** → Script runs on CPU by default; install CUDA-enabled PyTorch to use GPU

## 📄 License

MIT — feel free to use and adapt with attribution.

## 🙌 Acknowledgments

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [PyTorch & torchvision](https://pytorch.org/)
* [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/)
