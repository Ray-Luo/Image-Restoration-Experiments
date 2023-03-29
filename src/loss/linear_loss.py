from torch.nn import L1Loss


class LinearLoss:
    def __init__(self):
        self.loss = L1Loss()

    def __call__(self, pred, gt):
        return self.loss(pred, gt) * 4000.0
