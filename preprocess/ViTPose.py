import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./dataset"  # path to folder containing train/dev/test
SAVE_DIR = "./pose_features"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------
print("Loading models...")

# Person detector (RT-DETR)
person_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained(
    "PekingU/rtdetr_r50vd_coco_o365", device_map=device
)

# Pose estimator (VitPose)
pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
pose_model = VitPoseForPoseEstimation.from_pretrained(
    "usyd-community/vitpose-base-simple", device_map=device
)

# Replace head to get intermediate encoder features
class Identity(torch.nn.Module):
    def __call__(self, x, flip_pairs):
        return x

pose_model.head = Identity()

# ------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------

def detect_person_boxes(image):
    """Return bounding box (x, y, w, h) for the first detected person."""
    inputs = person_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = person_model(**inputs)
    results = person_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(image.height, image.width)]).to(device), threshold=0.3
    )[0]
    boxes = results["boxes"][results["labels"] == 0]
    if len(boxes) == 0:
        return None
    box = boxes[0].cpu().numpy()
    # VOC -> COCO format
    box[2] = box[2] - box[0]
    box[3] = box[3] - box[1]
    return box[np.newaxis, :]  # shape (1, 4)


def extract_pose_features(image, person_boxes):
    """Extract [D] feature for one image given person_boxes."""
    inputs = pose_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)
    with torch.no_grad():
        out = pose_model(**inputs)
        feats = out["heatmaps"]  # shape [1, 768, 16, 12]
        feats = feats.squeeze(0)  # [768, 16, 12]

        # Apply min + max pooling
        avg = torch.mean(feats, dim=[1, 2])
        maxv = torch.amax(feats, dim=[1, 2])
        combined = torch.cat([avg, maxv], dim=0)  # [1536]

    return combined.cpu().numpy()


def process_video_folder(video_path, save_path):
    """Process all frames in a single video folder and save [T, D] features."""
    frame_files = sorted(
        [f for f in os.listdir(video_path) if f.lower().endswith((".jpg", ".png"))]
    )

    last_features = None
    video_features = []

    for frame_name in frame_files:
        frame_path = os.path.join(video_path, frame_name)
        image = Image.open(frame_path).convert("RGB")

        boxes = detect_person_boxes(image)
        if boxes is None:
            # fallback: reuse last detected features
            if last_features is not None:
                video_features.append(last_features)
            continue

        features = extract_pose_features(image, boxes)
        video_features.append(features)
        last_features = features

    if not video_features and last_features is not None:
        video_features = [last_features]

    video_features = np.stack(video_features, axis=0)  # [T, D]
    np.save(save_path, video_features)


# ------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------

for split in ["train", "dev", "test"]:
    split_dir = os.path.join(DATA_DIR, split)
    save_split_dir = os.path.join(SAVE_DIR, split)
    os.makedirs(save_split_dir, exist_ok=True)

    for video_folder in tqdm(sorted(os.listdir(split_dir)), desc=f"Processing {split}"):
        video_path = os.path.join(split_dir, video_folder)
        if not os.path.isdir(video_path):
            continue

        save_path = os.path.join(save_split_dir, f"{video_folder}.npy")
        if os.path.exists(save_path):
            continue

        try:
            process_video_folder(video_path, save_path)
        except Exception as e:
            print(f"Error processing {video_folder}: {e}")
