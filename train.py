import os
import torch
import visdom
from termcolor import colored
from easydict import EasyDict as edict
from train_utils import get_model, parse_task_dictionary, train_vanilla, \
                           adjust_learning_rate, save_checkpoint
from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataset import UDE_Dataset

def train(viz, lr=1e-4, start_path=None, batch_size=8, Max_epoch=200):
    datasets = UDE_Dataset(path_src, path_edge, path_mask, path_target, transforms=transforms.ToTensor())
    TASKS = parse_task_dictionary(tasks_dictionary)
    model = get_model(TASKS, TASKS)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    if start_path:
        experiment = torch.load(start_path)
        model.load_state_dict(experiment['model_state'])
        optimizer.load_state_dict(experiment['optimizer_state'])

    print('Train on {} samples'.format(len(datasets)))
    train_loader = DataLoader(datasets, batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Main loop
    print(colored('Starting main loop', 'blue'))
    global_step = 0
    for epoch in range(Max_epoch):
        print(colored('Epoch %d/%d' % (epoch + 1, Max_epoch), 'red'))
        print(colored('-' * 10, 'yellow'))

        # Train
        print('Train ...')
        global_step = train_vanilla(train_loader, model, viz, optimizer, global_step, TASKS)
        # Adjust lr
        lr = adjust_learning_rate(lr, decay=0.5, optimizer=optimizer, cur_epoch=epoch, n_epochs=20)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Checkpoint
        if epoch % 10 == 0:
            print('Checkpoint ...')
            weights_fname = 'weights-%d.pth.tar' % epoch
            save_checkpoint(model.state_dict(), optimizer.state_dict(), os.path.join(save_path, weights_fname))

if __name__ == "__main__":
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    path_src = './datasets/train_dataset/Raw/'
    path_edge = 'The path to the truth file of the semantic edge/'# resize to（256*256）
    path_mask = 'The path of semantic segmentation mask/'# resize to（256*256）
    path_target = 'The path of the relative depth sample/'# eg:'/datasets/Target.pkl'

    save_path = 'The path of the output/'


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tasks_dictionary = edict({
                                'include_depth': True,
                                'include_edge': True,
                                'include_semseg': True
    })


    # Create the window and initialize
    viz = visdom.Visdom(port=8007)
    viz_win_name = ['initial_depth', 'middle_depth', 'depth', 'total',
                    'initial_edge', 'middle_edge', 'depth_edge',
                    'initial_semseg', 'semseg'
                    ]

    for name in viz_win_name:
        print(name)
        viz.line([0.], [0], win=name, opts=dict(title=name))
    train(viz, lr=1e-4, start_path=None, batch_size=8, Max_epoch=100)












