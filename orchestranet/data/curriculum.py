class OcclusionCurriculum:
    def __init__(self, total_epochs=100, max_prob=0.8, max_ratio=0.5):
        self.total_epochs = total_epochs
        self.max_prob = max_prob
        self.max_ratio = max_ratio

    def get_params(self, epoch):
        progress = min(1.0, epoch / max(1, (self.total_epochs * 0.8)))
        return {
            "occlusion_prob": progress * self.max_prob,
            "max_ratio": progress * self.max_ratio
        }

    def should_apply_occlusion(self, epoch):
        return epoch > 5 # Start occlusion after 5 epochs
