import torch
import numpy as np
import cv2

def Delta(pred, target, exp=1.0):
    maxRatio = torch.max(pred / target, target / pred)
    return (maxRatio < 1.25 ** exp).float().mean()

def Delta1(pred, target):
    maxRatio = torch.max(pred / target, target / pred)
    return (maxRatio < 1.25 ** 1).float().mean()

def extract_corners(predictions, N=None):
    def __get_contours(x, t):
        _, im2 = cv2.threshold(x, t, 255, cv2.THRESH_BINARY)
        # dilate the thresholded peaks to eliminate "pinholes"
        im3 = cv2.dilate(im2, None, iterations=2)
        contours, _ = cv2.findContours(im3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    corners = []
    predictions = predictions.detach().cpu()
    for pred in predictions:
        pred = pred.squeeze(0).numpy() * 255
        pred = pred.astype(np.uint8)
        H,W = pred.shape

        i = 254
        while i > 0:
            contours = __get_contours(pred, i)
            if N is None: break
            if len(contours) == N: break
            i -= 1

        for contour in contours:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            corners.append((cX / W, cY / H))
    assert N is None or len(corners) == N
    corners = sorted(corners, key=lambda x: x[0])
    return np.array(corners)

def corner_error(gt_cor_id, dt_cor_id, w, h):
    if not gt_cor_id.shape == dt_cor_id.shape: return np.sqrt(w**2 + h**2)
    mse = np.sqrt(((gt_cor_id - dt_cor_id)**2).sum(1)).mean()
    ce_loss = 100 * mse / np.sqrt(w**2 + h**2)
    return ce_loss

class CE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, gt):
        pass

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
        
        