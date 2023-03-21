
import torch
from torch import nn
import numpy as np
from torch.nn.modules.module import Module


class RelativeDepthLoss(nn.Module):
    def __init__(self):
        super(RelativeDepthLoss, self).__init__()

    def ranking_loss(self, z_A, z_B, target):
        """
        loss for a given set of pixels:
        z_A: predicted absolute depth for pixels A
        z_B: predicted absolute depth for pixels B
        ground_truth: Relative depth between A and B (-1, 1)
        """
        pred_depth = z_A - z_B
        log_loss = torch.mean(torch.log(1 + torch.exp(target[target != 0] * pred_depth[target != 0])))
        return log_loss

    def forward(self, output, target):
        total_loss = 0
        for index in range(len(output)):
            x_A = target['x_A'][index].long()
            y_A = target['y_A'][index].long()
            x_B = target['x_B'][index].long()
            y_B = target['y_B'][index].long()

            x_A = torch.clamp(x_A, 0, 255)
            y_A = torch.clamp(y_A, 0, 255)
            x_B = torch.clamp(x_B, 0, 255)
            y_B = torch.clamp(y_B, 0, 255)


            z_A = output[index][0][x_A, y_A]  # all "A" points
            z_B = output[index][0][x_B, y_B]  # all "B" points

            total_loss += self.ranking_loss(z_A, z_B, target['ordinal_relation'][index])

        return total_loss / len(output)


class BinaryCrossEntropyLoss(nn.Module):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    For semantic segmentation(one-hot)
    """

    def __init__(self, size_average=True, batch_average=True):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average

    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())

        labels = torch.ge(label, 0.5).float()

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)
        final_loss = loss_pos + loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss

class BalancedCrossEntropyLoss(Module):
    """
    Balanced Cross Entropy Loss with optional ignore regions
    For edge
    """

    def __init__(self, size_average=True, batch_average=True, pos_weight=None):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight

    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())
        labels = torch.ge(label, 0.5).float()

        # Weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None and not self.pos_weight:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
            w = num_labels_neg / num_total

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        final_loss = w * loss_pos + (1 - w) * loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss



def edge_loss(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    mask = (edges > 0.5).float()
    b, c, h, w = mask.shape
    num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
    num_neg = c * h * w - num_pos                     # Shape: [b,].
    weight = torch.zeros_like(mask)
    weight[edges > 0.5] = num_neg / (num_pos + num_neg)
    weight[edges <= 0.5] = num_pos / (num_pos + num_neg)
    # Calculate loss.
    losses = torch.nn.functional.binary_cross_entropy(preds.float(), edges.float(), weight=weight, reduction='none')
    loss = torch.sum(losses) / b
    return loss




