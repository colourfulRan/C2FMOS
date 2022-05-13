import numpy as np
import argparse
import os
import glob
import SimpleITK as sitk
import cv2
import math
import matplotlib.pyplot as plt



def if_out_of_bounds(local, shape):
    x, y, z = local
    h, w, c = shape
    h, w, c = h-1, w-1, c-1
    if(x<0 or y<0 or z<0 or x>h or y>w or z>c):
        return False
    else:
        return True


def erode(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dst = cv2.erode(image, kernel)
    return dst

def dilate(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dst = cv2.dilate(image, kernel)
    return dst


def convert_pointcloud(img_arr, proba_arr, label_arr, probability_threshold):

    # canny_edges_3d(img_arr)
    batch, n, m = img_arr.shape
    img_mark = np.zeros_like(img_arr)
    label_arr = label_arr.astype(np.uint8)
    proba_arr = np.where(proba_arr > probability_threshold, 1, 0)
    proba_arr = proba_arr.astype(np.uint8)
    def dfs(z, x, y, cnt):
        if cnt > 12:
            return
        if not 0 <= x < n or not 0 <= y < m or not 0 < img_arr[z, x, y] < 100:
            return
        img_mark[z, x, y] = 1
        if img_mark[z, x + 1, y] == 0:
            dfs(z, x + 1, y, cnt+1)

    for i in range(batch):

        image = proba_arr[i, :, :]
        img_mark[i] = image
        image_index = np.nonzero(image == 1)
        image_index = np.array(image_index[::-1])
        image_index = np.transpose(image_index)
        if len(image_index) > 1:
            for ids, (k, j) in enumerate(image_index):
                dfs(i, j, k, 0)
        image = img_mark[i, :, :]
        image = erode(image, 5)
        image = dilate(image, 7)
        img_mark[i] = image

    # get indices of probabilites larger some value
    # 构造点云，概率值大于阈值
    proba_indices = np.nonzero(img_mark == 1)  # proba_arr > 0.5
    proba_indices = np.array(proba_indices[::-1])
    proba_indices = np.transpose(proba_indices)  # shape is (N, 3)  矩阵转置

    indices = proba_indices
    labels = np.zeros((indices.shape[0], 1), np.uint8)  # shape is (N, 1)
    feature = np.zeros((indices.shape[0], 2), np.float32)

    for idx, (x, y, z) in enumerate(indices):
        labels[idx] = label_arr[z, y, x]
        feature[idx][0] = img_arr[z, y, x]
        feature[idx][1] = proba_arr[z, y, x]

    return indices, feature, labels



def create_pointcloud_data(dir: str, save_dir: str, threshhold: int = 0.9):
    """创建数据"""
    pro_dir=os.path.join(dir, 'probability')
    gt_dir=os.path.join(dir, 'label')
    image_dir = os.path.join(dir, 'image_original')
    save_pc = os.path.join(save_dir, 'scans')
    save_label = os.path.join(save_dir, 'labels')
    save_feature = os.path.join(save_dir, 'features_original')

    os.makedirs(save_pc, exist_ok=True)
    os.makedirs(save_label, exist_ok=True)
    os.makedirs(save_feature, exist_ok=True)

    scan_files = glob.glob(os.path.join(pro_dir, '*'))

    for scan_file in scan_files:
        scan_name = os.path.basename(scan_file)
        label_name = scan_name.replace("pred", "label")
        image_name = scan_name.replace('_pred', '')
        label_file = os.path.join(gt_dir, label_name)
        image_file = os.path.join(image_dir, image_name)
        labels = sitk.ReadImage(label_file)
        labels = sitk.GetArrayFromImage(labels)
        pro = sitk.ReadImage(scan_file)
        pro = sitk.GetArrayFromImage(pro)
        img = sitk.ReadImage(image_file)
        img = sitk.GetArrayFromImage(img)
        # area_grow(img)
        pc, feature, gt = convert_pointcloud(img, pro, labels, threshhold)
        print('{} : {}, {}'.format(scan_name, pc.shape, np.count_nonzero(gt==1)) )
        pc.tofile(os.path.join(save_pc, "{}.bin".format(scan_name.replace(".nii.gz", ""))))
        gt.tofile(os.path.join(save_label, "{}.label".format(scan_name.replace(".nii.gz", ""))))
        feature.tofile(os.path.join(save_feature, "{}.bin".format(scan_name.replace(".nii.gz", ""))))
        print("{} has completed".format(scan_name))




def main(data_dir: str, save_dir:str):

    # IQ 的阈值为概率0.5
    probability_threshold = 0.5

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # let's create some sample data
    np.random.seed(42)  # to have same sample data
    create_pointcloud_data(data_dir, save_dir, probability_threshold)




def preprocess_data():
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Dataset creation')

    parser.add_argument(
        '--data_dir',
        type=str,
        default='tmp/',
        help='Path to the data directory.'
    )


    parser.add_argument(
        '--save_dir',
        type=str,
        default='tmp/pointcloud_0.5',
        help='Path to save subject'
    )

    args = parser.parse_args()
    main(args.data_dir, args.save_dir)
