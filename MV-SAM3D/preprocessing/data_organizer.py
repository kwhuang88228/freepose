"""
数据组织模块 - 将原始图片整理到 images/ 目录
"""
import shutil
import cv2
from pathlib import Path
from typing import List, Dict
from loguru import logger


def organize_images(scene_dir: Path) -> Dict:
    """
    将散乱的图片整理到 images/ 目录
    
    Args:
        scene_dir: 场景目录（包含 0.jpg, 1.jpg 等）
    
    Returns:
        Dict with status and info
    """
    logger.info(f"Organizing images in: {scene_dir}")
    
    # 查找原始图片
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(scene_dir.glob(ext)))
    
    # 过滤掉已经在 images/ 目录中的
    image_files = [f for f in image_files if 'images' not in f.parts]
    
    if not image_files:
        # 检查是否已经有 images/ 目录
        images_dir = scene_dir / "images"
        if images_dir.exists():
            existing_images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            if existing_images:
                logger.info(f"  Images already organized ({len(existing_images)} files)")
                return {
                    'success': True,
                    'num_images': len(existing_images),
                    'already_organized': True,
                }
        
        logger.error("  No image files found!")
        return {'success': False, 'error': 'No images found'}
    
    # 按文件名排序（数字顺序）
    def natural_sort_key(p):
        try:
            return int(p.stem)
        except:
            return p.stem
    
    image_files = sorted(image_files, key=natural_sort_key)
    logger.info(f"  Found {len(image_files)} images")
    
    # 创建 images/ 目录
    images_dir = scene_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # 复制并重命名图片
    for i, img_path in enumerate(image_files):
        output_path = images_dir / f"{i}.png"
        
        # 读取并保存为 PNG
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"  Failed to read: {img_path}")
            continue
        
        cv2.imwrite(str(output_path), img)
        logger.info(f"  {img_path.name} → {output_path.name}")
    
    logger.success(f"✓ Organized {len(image_files)} images to {images_dir}")
    
    return {
        'success': True,
        'num_images': len(image_files),
        'images_dir': images_dir,
        'already_organized': False,
    }
