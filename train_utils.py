'''
Function: Customize the function used in the training process
'''
import torch
import torch.nn as nn
from termcolor import colored
from torch.autograd import Variable
from shutil import copyfile
import torch.nn.functional as F
from easydict import EasyDict as edict
from losses.criterion import RelativeDepthLoss, BinaryCrossEntropyLoss, BalancedCrossEntropyLoss


def get_model(TASKS, AUXILARY_TASKS):
    """ Return the model """
    from models.seg_hrnet import hrnet_w18
    backbone = hrnet_w18(pretrained=False)
    backbone_channels = [18, 36, 72, 144]
    from models.seg_hrnet import HighResolutionFuse
    backbone = torch.nn.Sequential(backbone, HighResolutionFuse(backbone_channels, 256))
    backbone_channels = sum(backbone_channels)

    from models.model import UDE_Net
    model = UDE_Net(TASKS, AUXILARY_TASKS, backbone, backbone_channels)
    return model

def parse_task_dictionary(task_dic):
    """
        Return a dictionary with task information.
        Additionally we return a dict with key, values to be added to the main dictionary
    """

    task_cfg = edict()
    task_cfg.NAMES = []
    task_cfg.NUM_OUTPUT = {}
    if 'include_semseg' in task_dic.keys() and task_dic['include_semseg']:
        # Semantic segmentation
        tmp = 'semseg'
        task_cfg.NAMES.append('semseg')
        task_cfg.NUM_OUTPUT[tmp] = 8
    if 'include_depth' in task_dic.keys() and task_dic['include_depth']:
        # Depth
        tmp = 'depth'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
    if 'include_edge' in task_dic.keys() and task_dic['include_edge']:
        # edge
        tmp = 'edge'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1

    return task_cfg

class Edge_Detect(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, bias=False, padding=1).cuda()
        self.conv2 = nn.Conv2d(1, 1, 3, bias=False, padding=1).cuda()

    def forward(self, tensor):
        k1 = torch.tensor([
            [-1.0, 0, 1.0],
            [-2.0, 0, 2.0],
            [-1.0, 0, 1.0]])

        k2 = torch.tensor([
            [1.0, 2.0, 1.0],
            [0, 0, 0],
            [-1.0, -2.0, -1.0]])
        k1 = k1.reshape((1, 1, 3, 3))
        k2 = k2.reshape((1, 1, 3, 3))
        self.conv1.weight.data = k1.cuda()
        self.conv2.weight.data = k2.cuda()

        conv2d_x = self.conv1(tensor.cuda())
        conv2d_y = self.conv2(tensor.cuda())
        x = conv2d_x ** 2
        y = conv2d_y ** 2
        edge = torch.sqrt(torch.add(x, y) + (1e-8))
        return edge

""" 
    Loss functions 
"""
def compute_losses(semseg, rank, edge_, pred, task_dic):
    total = 0.
    out = {}
    img_size = (256, 256)
    tasks = task_dic.NAMES

    # loss
    loss_semseg = BinaryCrossEntropyLoss()
    loss_BCE = BalancedCrossEntropyLoss()
    loss_R = RelativeDepthLoss()
    edge_detect = Edge_Detect()


    if 'depth' in tasks:
        # depth
        pred_depth = F.interpolate(pred['initial_depth'], img_size, mode='bilinear')
        loss_depth = loss_R(pred_depth, rank)
        out['initial_depth'] = loss_depth

        loss_depth_out = loss_R(F.interpolate(pred['middle_depth'], img_size, mode='bilinear'), rank)
        out['middle_depth'] = loss_depth_out

        loss_depth2_out = loss_R(pred['depth'], rank)
        out['depth'] = loss_depth2_out

        total = loss_depth + loss_depth2_out + loss_depth_out

    if 'edge' in tasks:
        # edge
        pred_edge = F.interpolate(pred['initial_edge'], img_size, mode='bilinear')
        loss_edge = 5 * loss_BCE(pred_edge, edge_)
        out['initial_edge'] = loss_edge
        # out_edge
        loss_edge_out = 5*loss_BCE(pred['middle_edge'], edge_)
        out['middle_edge'] = loss_edge_out

        loss_out_edge2 = 5*loss_BCE(edge_detect(pred['depth']), edge_)
        out['depth_edge'] = loss_out_edge2

        total = total + loss_edge + loss_edge_out + loss_out_edge2

    if 'semseg' in tasks:
        # semsge
        pred_semseg = F.interpolate(pred['initial_semseg'], img_size, mode='bilinear')
        loss_sem = 5*loss_semseg(pred_semseg, semseg)
        out['initial_semseg'] = loss_sem

        loss_sem_out = 5*loss_semseg(pred['middle_semseg'], semseg)
        out['semseg'] = loss_sem_out

        total = total + loss_sem + loss_sem_out

    out['total'] = total

    return out


def train_vanilla(train_loader, model, viz, optimizer, steps, tasks_dict):
    """ Vanilla training with fixed loss weights """
    i = 0
    model.train()
    for img, edge, mask, target in train_loader:
        img = Variable(img.cuda())
        edge = Variable(edge.cuda())
        mask = Variable(mask.cuda())
        target['x_A'] = target['x_A'].cuda()
        target['y_A'] = target['y_A'].cuda()
        target['x_B'] = target['x_B'].cuda()
        target['y_B'] = target['y_B'].cuda()
        target['ordinal_relation'] = Variable(target['ordinal_relation']).cuda()

        output = model(img)
        # Measure loss
        loss_dict = compute_losses(mask, target, edge, output, tasks_dict)

        i += 1
        if i % 20 == 0:
            steps += 1
            viz.images(img, win='img')
            viz.images(output['depth'], win='pre_depth')
            print(colored('-' * 40, 'blue'))
            for key in loss_dict:
                print(key + ':', loss_dict[key])
                viz.line([float(loss_dict[key])], [steps], win=key, update='append')
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()
    return steps

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def save_checkpoint(model_state, optimizer_state, filename, epoch=None, is_best=False):
    state = dict(model_state=model_state,
                 optimizer_state=optimizer_state,
                 epoch=epoch)
    torch.save(state, filename)
    if is_best:
        copyfile(filename, 'model_best.pth.tar')



