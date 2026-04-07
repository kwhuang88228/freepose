import cv2
from pathlib import Path
from tqdm import tqdm

def frames_to_video(frames_dir: Path, output_path: Path, fps: int = 30) -> None:
    """Create a video from a directory of frames."""
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise ValueError(f"No frames found in {frames_dir}")
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(str(frames[0]))
    height, width, _ = first_frame.shape
    
    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    root_dir = Path("/share/hariharan/kh775/code/freepose/data/results/hd_epic_clips/v2")

    for video_dir in tqdm(root_dir.iterdir()):
        # frames_dir = root_dir / video_dir / f"viz_bbox_{video_dir.name}-tracked"
        frames_dir = root_dir / video_dir / "sam2_masks"
        output_path = root_dir / video_dir / f"{video_dir.name}_mask.mp4"
        frames_to_video(frames_dir, output_path)