from torch.nn import L1Loss


class Normalizationloss:
    def __init__(self):
        self.loss = L1Loss()

    def denormalize(self, x):
        return x * 4000.0

    def __call__(self, pred, gt):
        return self.loss(self.denormalize(pred), self.denormalize(gt))
