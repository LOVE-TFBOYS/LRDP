import torch

from losses.regularization import jacobian_determinant_3d


def dice_score_per_label(pred, target, labels=None, include_background=False, eps=1e-5):
    pred = pred.long()
    target = target.long()
    if labels is None:
        labels = torch.unique(torch.cat([pred.reshape(-1), target.reshape(-1)]))
    scores = []
    for label in labels:
        label_value = int(label.item()) if torch.is_tensor(label) else int(label)
        if not include_background and label_value == 0:
            continue
        pred_mask = pred == label_value
        target_mask = target == label_value
        if not pred_mask.any() and not target_mask.any():
            continue
        intersection = (pred_mask & target_mask).float().sum()
        denominator = pred_mask.float().sum() + target_mask.float().sum()
        scores.append((2.0 * intersection + eps) / (denominator + eps))
    if not scores:
        return pred.new_tensor(1.0, dtype=torch.float32)
    return torch.stack(scores).mean()


def batch_dice_score(pred, target, include_background=False):
    scores = []
    for index in range(pred.shape[0]):
        scores.append(dice_score_per_label(pred[index], target[index], include_background=include_background))
    return torch.stack(scores).mean()


def folding_ratio(flow):
    """Ratio of non-positive Jacobian determinants, detJ <= 0."""

    jacobian = jacobian_determinant_3d(flow)
    return (jacobian <= 0).float().mean()
