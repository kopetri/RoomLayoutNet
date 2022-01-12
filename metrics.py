import torch

def Delta(pred, target, exp=1.0):
    maxRatio = torch.max(pred / target, target / pred)
    return (maxRatio < 1.25 ** exp).float().mean()

def Delta1(pred, target):
    maxRatio = torch.max(pred / target, target / pred)
    return (maxRatio < 1.25 ** 1).float().mean()

class IoU(torch.nn.Module):
    def __init__(self, alpha=0.99) -> None:
        super().__init__()
        self.alpha = alpha

    def compute_corner_mask(self, x):
        corner_area = x >= self.alpha
        mask = torch.zeros_like(x)
        mask[corner_area] = 1.0
        mask[~corner_area] = 0.0
        return mask

    def forward(self, pred, gt):
        mask_pred = self.compute_corner_mask(pred)
        mask_gt   = self.compute_corner_mask(gt)
        intersection = mask_pred * mask_gt
        union = torch.clip(mask_pred + mask_gt, 0, 1)
        return torch.sum(intersection) / torch.sum(union)
        
        