from .coco import CocoDataset

class PasteCOCO(CocoDataset):
    def __init__(self, extra_cls, **kwargs):
        self.extra_cls = extra_cls
        super(CocoDataset, self).__init__(**kwargs)