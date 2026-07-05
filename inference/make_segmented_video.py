import cv2
import tempfile
from pathlib import Path
from tqdm import tqdm

from frames_to_video import frames_to_video

frames_dir = Path("/share/hariharan/kh775/code/freepose/data/datasets/videos/P01-20240202-110250_3_knife")
masks_dir = Path("/share/hariharan/kh775/code/freepose/data/results/mvsam3d_groundingdino_sam2/P01-20240202-110250_3_knife/01_detection_tracking/tracking/binary_masks")
output_path = Path("/share/hariharan/kh775/code/freepose/data/results/mvsam3d_groundingdino_sam2/P01-20240202-110250_3_knife/01_detection_tracking/tracking/object_segmented.mp4")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for frame_path in tqdm(sorted(frames_dir.glob("*.png"))):
        mask_path = masks_dir / f"{frame_path.stem}_mask.png"
        frame = cv2.imread(str(frame_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        segmented = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imwrite(str(tmp / frame_path.name), segmented)

    frames_to_video(tmp, output_path, fps=30)

print(f"Wrote {output_path}")
