import cv2
import glob
import os
from pathlib import Path
import subprocess
from tqdm import tqdm


def frames_to_video(frames_dir, output_path, fps=16):

    frames_pattern = os.path.join(str(frames_dir), "*.png")
    if not glob.glob(frames_pattern):
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-pattern_type", "glob",
            "-i", frames_pattern,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(output_path)],
        check=True,
    )


if __name__ == "__main__":
    # root_dir = Path("/share/hariharan/kh775/code/freepose/data/results/hd_epic_clips/v2")

    # for video_dir in tqdm(root_dir.iterdir()):
    #     # frames_dir = root_dir / video_dir / f"viz_bbox_{video_dir.name}-tracked"
    #     frames_dir = root_dir / video_dir / "sam2_masks"
    #     output_path = root_dir / video_dir / f"{video_dir.name}_mask.mp4"
    #     frames_to_video(frames_dir, output_path)

    # root_dir = Path("/share/hariharan/kh775/code/freepose/data/results/mvsam3d/masked_unmasked_masked_unmasked")

    # for video_dir in tqdm(root_dir.iterdir()):
    #     frames_dir = root_dir / video_dir / "04_coarse_poses" / "bbox3d"
    #     output_path = root_dir / video_dir / f"{video_dir.name}_coarse_poses_bbox3d.mp4"
    #     frames_to_video(frames_dir, output_path)
    #     # frames_dir = root_dir / video_dir / "04_coarse_poses" / "gaussian"
    #     # output_path = root_dir / video_dir / f"{video_dir.name}_coarse_poses_gaussian.mp4"
    #     # frames_to_video(frames_dir, output_path)
    #     # frames_dir = root_dir / video_dir / "05_tracked" / "bbox3d"
    #     # output_path = root_dir / video_dir / f"{video_dir.name}_tracked_bbox3d.mp4"
    #     # frames_to_video(frames_dir, output_path)
    #     # frames_dir = root_dir / video_dir / "05_tracked" / "cotracker"
    #     # output_path = root_dir / video_dir / f"{video_dir.name}_tracked_cotracker.mp4"
    #     # frames_to_video(frames_dir, output_path)

    root_dir = Path("/share/hariharan/kh775/code/freepose/data/results/mvsam3d_groundingdino_sam2")
    for video_dir in tqdm(root_dir.iterdir()):
        video_dir = root_dir / video_dir / "01_detection_tracking" / "tracking"
        masks_dir = video_dir / "binary_masks"
        boxes_dir = video_dir / "boxes"
        masks_overlay_dir = video_dir / "masks_overlay"

        frames_to_video(masks_dir, video_dir / f"binary_masks.mp4")
        frames_to_video(boxes_dir, video_dir / f"boxes.mp4")
        frames_to_video(masks_overlay_dir, video_dir / f"masks_overlay.mp4")
        
        
        
    