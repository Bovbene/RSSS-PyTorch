import argparse
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.deeplabv3_version_1.deeplabv3 import DeepLabV3
from torch.autograd import Variable
import torch
#import os
#import pandas as pd
from PIL import Image
from collections import OrderedDict
import torch.nn as nn
from torchvision import transforms
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

#====================================================================================================
import cv2 as cv
def GetPadImNRowColLi(image_path,cutsize_h = 512,cutsize_w = 512,stride = 512):
    image = cv.imread(image_path)
    h,w = image.shape[0],image.shape[1]
    h_pad_cutsize = h if (h // cutsize_h == 0) else (h // cutsize_h + 1) * cutsize_h
    w_pad_cutsize = w if (w // cutsize_w == 0) else (w // cutsize_w + 1) * cutsize_w
    image = cv.copyMakeBorder(image,
                              0,
                              h_pad_cutsize - h,
                              0, 
                              w_pad_cutsize - w, 
                              cv.BORDER_CONSTANT, 0)
    N = image.shape[0]-cutsize_h+1
    M = image.shape[1]-cutsize_w+1
    from numpy import arange
    row = arange(0,N,stride)
    col = arange(0,M,stride)
    row_col_li = []
    for c in col:
        for r in row:
            row_col_li.append([c,r,c+cutsize_w,r+cutsize_h])
    return image,row_col_li
#====================================================================================================


def snapshot_forward(model, dataloader, model_list, png,shape):
    model.eval()
    for (index, (image, pos_list)) in enumerate(dataloader):
        image = Variable(image).cuda()
        # print(image)
        # print(pos_list)

        predict_list = 0
        for model in model_list:
            predict_1 = model(image)
            # predict_list = predict_1
            predict_2 = model(torch.flip(image, [-1]))
            predict_2 = torch.flip(predict_2, [-1])

            predict_3 = model(torch.flip(image, [-2]))
            predict_3 = torch.flip(predict_3, [-2])

            predict_4 = model(torch.flip(image, [-1, -2]))
            predict_4 = torch.flip(predict_4, [-1, -2])

            predict_list += (predict_1 + predict_2 + predict_3 + predict_4)
        predict_list = torch.argmax(predict_list.cpu(), 1).byte().numpy()  # n x h x w

        batch_size = predict_list.shape[0]  # batch大小
        for i in range(batch_size):
            predict = predict_list[i]
            pos = pos_list[i, :]
            [topleft_x, topleft_y, buttomright_x, buttomright_y] = pos

            if (buttomright_x - topleft_x) == 512 and (buttomright_y - topleft_y) == 512:
                #png[topleft_y + 128:buttomright_y - 128, topleft_x + 128:buttomright_x - 128] = predict[128:384,128:384]
                png[topleft_y:buttomright_y, topleft_x:buttomright_x] = predict
            else:
                raise ValueError(
                    "target_size!=512， Got {},{}".format(buttomright_x - topleft_x, buttomright_y - topleft_y))
    
    h, w = png.shape
    #png = png[128:h - 128, 128:w - 128]  # 去除整体外边界
    #zeros = (6800, 7200)  # 去除补全512整数倍时的右下边界
    zeros = shape
    png = png[:zeros[0], :zeros[1]]
    
    return png
def parse_args():
    parser = argparse.ArgumentParser(description="膨胀预测")
    parser.add_argument('--test-data-root', type=str, default='./data/image1/GF2_PMS2__20160225_L1A0001433318-MSS2.jpg')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N', help='batch size for testing (default:16)')
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument("--model-path", type=str, default='./fifteen/deeplabv3_version_1/resnet50/epoch_11.pth')
    parser.add_argument("--pred-path", type=str, default="")
    args = parser.parse_args()
    return args

def create_png(shape):
    #zeros = (6800, 7200)
    zeros = shape
    h, w = zeros[0], zeros[1]
    new_h = h if (h//512 == 0) else (h//512+1)*512
    new_w = w if (w//512 == 0) else (w//512+1)*512
    #new_h, new_w = (h//512+1)*512, (w//512+1)*512 # 填充下边界和右边界得到滑窗的整数倍
    #zeros = (new_h+128, new_w+128)  # 填充空白边界，考虑到边缘数据
    zeros = (new_h, new_w)
    zeros = np.zeros(zeros, np.uint8)
    return zeros

#====================================================================================================
class Inference_Dataset(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        #self.csv_file = pd.read_csv(csv_file, header=None)
        self.pad_image,self.row_col_li = GetPadImNRowColLi(root_dir)
        self.transforms = transforms

    def __len__(self):
        #return len(self.csv_file)
        return len(self.row_col_li)

    def __getitem__(self, idx):
        c,r,c_end,r_end = self.row_col_li[idx]
        image = Image.fromarray(self.pad_image[r:r_end,c:c_end])
        image = self.transforms(image)
        pos_list = np.array(self.row_col_li[idx])
        return image, pos_list
        '''
        filename = self.csv_file.iloc[idx, 0]
        # print(filename)
        image_path = os.path.join(self.root_dir, filename)
        # image = np.asarray(Image.open(image_path))  # mode:RGBA
        # image = cv.cvtColor(image, cv.COLOR_RGBA2BGRA)  # PIL(RGBA)-->cv2(BGRA)
        image = Image.open(image_path).convert('RGB')

        # if self.transforms:
        #     print('transforms')
        image = self.transforms(image)

        pos_list = self.csv_file.iloc[idx, 1:].values.astype("int")  # ---> (topleft_x,topleft_y,buttomright_x,buttomright_y)
        return image, pos_list
        '''
#====================================================================================================



def reference():
    args = parse_args()

    dataset = Inference_Dataset(root_dir=args.test_data_root,
                                transforms=img_transform)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    model = DeepLabV3(num_classes=16)
    state_dict = torch.load(args.model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model_list = []
    model_list.append(model)
    
    #==================================================================
    shape = cv.imread(args.test_data_root).shape
    zeros = create_png((shape[0],shape[1]))
    image = snapshot_forward(model, dataloader, model_list, zeros,(shape[0],shape[1]))
    #==================================================================
    label_save_path = "vis_image_" + "_predict.png"
    from palette import colorize_mask
    overlap=colorize_mask(image)
    overlap.save(label_save_path)


if __name__ == '__main__':
    reference()
