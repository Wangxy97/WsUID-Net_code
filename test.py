import os
import torch
from easydict import EasyDict as edict
from torchvision import transforms
from data.dataset import TEST_Dataset
from torch.autograd import Variable
from train_utils import get_model, parse_task_dictionary

def prep_img(img):
    return Variable(img.unsqueeze(0)).cuda()

def show_img(tensor):
    to_pil = transforms.ToPILImage()
    img = to_pil((tensor - tensor.min()) / (tensor.max() - tensor.min()))  # min/max scaling
    return img


def test(ckpt, model, optimizer, test_dataset, test_path, out_path):
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    model.eval()
    filename_list = os.listdir(test_path)
    for i in range(0, len(filename_list)):
        filename = filename_list[i]
        output = model(prep_img(test_dataset[i]))
        depth = show_img(output['depth'].data.cpu()[0])
        depth.save(out_path + filename)
        print("Save image {}/{}".format(i, len(filename_list)))
    print('finish')

if __name__ == '__main__':
    ckpt_path = './out/checkpoint/weights.pth.tar'
    test_img = '/datasets/test_dataset/Raw/'
    out_path = './out/result/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    test_dataset = TEST_Dataset(test_img, transforms=transforms.ToTensor())
    tasks_dictionary = edict({
                                'include_depth': True,
                                'include_edge': True,
                                'include_semseg': True
                                       })

    main_tasks = parse_task_dictionary(tasks_dictionary)
    model = get_model(main_tasks, main_tasks)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    test(ckpt_path, model, optimizer, test_dataset, test_img, out_path)