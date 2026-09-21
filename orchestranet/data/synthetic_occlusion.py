import random
from PIL import ImageDraw

class SyntheticOcclusionGenerator:
    def __init__(self, max_occlusion_ratio=0.5):
        self.max_occlusion_ratio = max(0.0, float(max_occlusion_ratio))

    def __call__(self, img, boxes):
        # A simple placeholder occlusion generator that draws black boxes over image
        # This will simulate occlusion for training
        if random.random() < 0.5 or self.max_occlusion_ratio <= 0.0:
            return img, boxes
            
        draw = ImageDraw.Draw(img)
        w, h = img.size

        max_w = max(11, int(w * self.max_occlusion_ratio))
        max_h = max(11, int(h * self.max_occlusion_ratio))

        for _ in range(random.randint(1, 3)):
            box_w = random.randint(10, min(w, max_w))
            box_h = random.randint(10, min(h, max_h))
            x = random.randint(0, max(0, w - box_w))
            y = random.randint(0, max(0, h - box_h))
            draw.rectangle([x, y, x + box_w, y + box_h], fill=(0, 0, 0))
            
        return img, boxes
