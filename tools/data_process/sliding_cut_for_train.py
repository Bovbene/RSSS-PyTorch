import cv2
import os


def preprocess(path):
    """
    rgb: 原始图像
    lab：标签
    将图像裁剪为512*512大小，stride为256
    :return:
    """
    rgb = path+'/rgb/'
    label = path+'/label/'
    img_list = os.listdir(rgb)
    lab_list = os.listdir(label)
    img_files = [img.split('.')[0] for img in img_list]
    lab_files = [lab.split('.')[0] for lab in lab_list]
    flag = 1
    img_folder = path+'/patch_img/'
    lab_folder = path+'patch_lab/'
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    if not os.path.exists(lab_folder):
        os.makedirs(lab_folder)
    for i in range(len(img_files)):
        if img_files[i] not in lab_files:
            continue

        img = cv2.imread(rgb+img_list[i])
        lab = cv2.imread(label+lab_list[i])
        h, w, c = img.shape
        for h_ in range(0, h-512, 256):
            for w_ in range(0, w-512, 256):
                img_ = img[h_:h_+512, w_:w_+512, :]
                lab_ = lab[h_:h_ + 512, w_:w_ + 512, :]
                img_name = img_folder + str(flag) + '.png'
                lab_name = lab_folder + str(flag) + '.png'
                cv2.imwrite(img_name, img_)
                cv2.imwrite(lab_name, lab_)
                flag += 1

