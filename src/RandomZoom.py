from typing import List, Tuple

import numpy as np
from detectron2.data import transforms as T


class RandomZoom(T.Augmentation):
    def __init__(self, zooms: List[Tuple[float, float]], prob=0.5):
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        self.zooms = zooms
        self.prob = prob
        self.transforms = [T.RandomApply(T.RandomCrop("relative", (x, y)), prob=1) for x,y in zooms]

    def get_transform(self, img):
        do = self._rand_range() < self.prob
        if do:
            transform = np.random.choice(self.transforms)
            return transform.get_transform(img)
        else:
            return T.NoOpTransform()