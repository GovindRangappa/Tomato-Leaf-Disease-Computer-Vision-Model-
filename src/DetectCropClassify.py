import os
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# === Config ===
HF_REPO = "GovRang/tomato-leaf-model"  # <- your HF repo

YOLO_MODEL_PATH = hf_hub_download(HF_REPO, "best.pt")
CLASSIFIER_MODEL_PATH = hf_hub_download(HF_REPO, "merged_model.pth")

INPUT_IMAGE_PATH = "examples/DetectionTest.jpg"
OUTPUT_DIR = 'detected_and_classified'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Initialize Models ===
detector = YOLO(YOLO_MODEL_PATH)

classifier = models.resnet18(weights=None)
classifier.fc = torch.nn.Linear(classifier.fc.in_features, 3)  # ✅ Changed to 3 classes
classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location='cpu'))
classifier.eval()

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class_names = ['Bacterial_Spot', 'Early_Blight', 'Healthy']  # ✅ Updated class names

# === Detection ===
results = detector(INPUT_IMAGE_PATH)[0]
original_image = cv2.imread(INPUT_IMAGE_PATH)

for i, box in enumerate(results.boxes.xyxy.cpu().numpy()):
    x1, y1, x2, y2 = map(int, box)
    conf = float(results.boxes.conf[i])

    crop = original_image[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    # Save cropped image
    crop_path = os.path.join(OUTPUT_DIR, f'crop_{i}_conf_{conf:.2f}.jpg')
    cv2.imwrite(crop_path, crop)

    # Classify the crop
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    input_tensor = transform(crop_pil).unsqueeze(0)

    with torch.no_grad():
        outputs = classifier(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_conf, pred_class = torch.max(probs, 1)

    label = f"{class_names[pred_class]} ({pred_conf.item():.2f})"
    print(f"Detected leaf {i}: {label}")

    # Annotate original image
    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(original_image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save annotated original image
cv2.imwrite(os.path.join(OUTPUT_DIR, 'annotated_image.jpg'), original_image)

print(f"All crops and annotated image saved in '{OUTPUT_DIR}'")
