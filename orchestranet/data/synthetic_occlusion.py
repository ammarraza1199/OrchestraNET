import random
from PIL import ImageDraw

class SyntheticOcclusionGenerator:
    def __init__(self, max_occlusion_ratio=0.5):
        self.max_occlusion_ratio = max_occlusion_ratio

    def __call__(self, img, boxes):
        # A simple placeholder occlusion generator that draws black boxes over image
        # This will simulate occlusion for training
        if random.random() < 0.5:
            return img, boxes
            
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        for _ in range(random.randint(1, 3)):
            box_w = random.randint(10, int(w * self.max_occlusion_ratio))
            box_h = random.randint(10, int(h * self.max_occlusion_ratio))
            x = random.randint(0, w - box_w)
            y = random.randint(0, h - box_h)
            draw.rectangle([x, y, x + box_w, y + box_h], fill=(0, 0, 0))
            
        return img, boxes
