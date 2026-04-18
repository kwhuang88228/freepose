import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, model_name: str = 'dinov2_vitl14_reg'):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, images, layer=22, feature_type='cls'):
        with torch.inference_mode():
            # DINOv2 requires spatial dims to be multiples of patch_size (14).
            # Snap down to the nearest valid size to avoid the assertion error.
            patch_size = self.model.patch_embed.patch_size[0]
            h, w = images.shape[-2:]
            new_h = (h // patch_size) * patch_size
            new_w = (w // patch_size) * patch_size
            if new_h != h or new_w != w:
                images = F.interpolate(images, size=(new_h, new_w),
                                       mode='bilinear', align_corners=False)
            x = self.model.prepare_tokens_with_masks(self.transform(images), None)

            for blk_idx, blk in enumerate(self.model.blocks):
                x = blk(x)
                if blk_idx + 1 == layer:
                    break

            x_norm = self.model.norm(x)

            if feature_type == 'cls':
                out = x_norm[:, 0]
            elif feature_type == 'reg':
                out = x_norm[:, 1 : self.model.num_register_tokens + 1]
            elif feature_type == 'patch':
                out = x_norm[:, self.model.num_register_tokens + 1 :]
                
            return out

