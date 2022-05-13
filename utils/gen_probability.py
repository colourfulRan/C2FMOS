import argparse
import os, sys

from data_utils.MOTSDataset import MOTSValDataSet
from model.unet3D_DynConv882 import UNet3D
from utils.engine import Engine

sys.path.append("..")

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter

import timeit

import nibabel as nib
from math import ceil



start = timeit.default_timer()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MOTS: DynConv solution!")

    parser.add_argument("--data_dir", type=str, default='../tmp')
    parser.add_argument("--val_list", type=str, default='/MOTS_test.txt')
    parser.add_argument("--reload_path", type=str, default='logs/MOTS_DynConv_checkpoint_v1.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--save_path", type=str, default='tmp/test/')

    parser.add_argument("--input_size", type=str, default='64,192,192')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    return parser



def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()



def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def multi_net(net_list, img, task_id):
    # img = torch.from_numpy(img).cuda()

    padded_prediction = net_list[0](img, task_id)
    padded_prediction = F.sigmoid(padded_prediction)
    for i in range(1, len(net_list)):
        padded_prediction_i = net_list[i](img, task_id)
        padded_prediction_i = F.sigmoid(padded_prediction_i)
        padded_prediction += padded_prediction_i
    padded_prediction /= len(net_list)
    return padded_prediction#.cpu().data.numpy()

def predict_sliding(args, net_list, image, tile_size, classes, task_id):  # tile_size:32x256x256
    gaussian_importance_map = _get_gaussian(tile_size, sigma_scale=1. / 8)
    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
    
    image_size = image.shape
    overlap = 1 / 2

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    # print("Need %i x %i x %i prediction tiles @ stride %i x %i px" % (tile_deps, tile_cols, tile_rows, strideD, strideHW))
    full_probs = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)  # 1x4x155x240x240
    count_predictions = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    full_probs = torch.from_numpy(full_probs)
    count_predictions = torch.from_numpy(count_predictions)

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)  # for portrait images the x1 underflows sometimes
                y1 = max(int(y2 - tile_size[1]), 0)  # for very few rows y1 underflows

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img = torch.from_numpy(img).cuda()

                prediction1 = multi_net(net_list, img, task_id)
                prediction2 = torch.flip(multi_net(net_list, torch.flip(img, [2]), task_id), [2])
                prediction3 = torch.flip(multi_net(net_list, torch.flip(img, [3]), task_id), [3])
                prediction4 = torch.flip(multi_net(net_list, torch.flip(img, [4]), task_id), [4])
                prediction5 = torch.flip(multi_net(net_list, torch.flip(img, [2,3]), task_id), [2,3])
                prediction6 = torch.flip(multi_net(net_list, torch.flip(img, [2,4]), task_id), [2,4])
                prediction7 = torch.flip(multi_net(net_list, torch.flip(img, [3,4]), task_id), [3,4])
                prediction8 = torch.flip(multi_net(net_list, torch.flip(img, [2,3,4]), task_id), [2,3,4])
                prediction = (prediction1 + prediction2 + prediction3 + prediction4 + prediction5 + prediction6 + prediction7 + prediction8) / 8.
                prediction = prediction.cpu()
                # prediction = prediction.numpy()
                
                prediction[:,:] *= gaussian_importance_map

                if isinstance(prediction, list):
                    shape = np.array(prediction[0].shape)
                    shape[0] = prediction[0].shape[0] * len(prediction)
                    shape = tuple(shape)
                    preds = torch.zeros(shape).cuda()
                    bs_singlegpu = prediction[0].shape[0]
                    for i in range(len(prediction)):
                        preds[i * bs_singlegpu: (i + 1) * bs_singlegpu] = prediction[i]
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += preds

                else:
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += gaussian_importance_map
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs

def save_nii(args, pred, label, image, name, affine): # bs, c, WHD
    seg_pred_2class = np.asarray(pred)
    seg_pred_0 = seg_pred_2class[:, 0, :, :, :]
    seg_pred_1 = seg_pred_2class[:, 1, :, :, :]
    seg_pred = np.zeros_like(seg_pred_0)
    if name[0][0:3] == 'lun' or name[0][0:3] == 'col':
        seg_pred = seg_pred_1
    elif name[0][0:3]=='spl':
        seg_pred = seg_pred_0
    image = np.squeeze(image.numpy(), axis=1)

    if name[0][0:3]!='cas':
        seg_pred_0 = seg_pred_0.transpose((0, 2, 3, 1))
        seg_pred_1 = seg_pred_1.transpose((0, 2, 3, 1))
        seg_pred = seg_pred.transpose((0, 2, 3, 1))
        image = image.transpose((0, 2, 3, 1))

    # save
    if name[0][0:3] == 'lun' or name[0][0:3] == 'col' or name[0][0:3]=='spl':
        for tt in range(image.shape[0]):
            seg_pred_tt = seg_pred[tt]
            image_tt = image[tt]
            seg_pred_tt = nib.Nifti1Image(seg_pred_tt, affine=affine[tt])
            image_tt = nib.Nifti1Image(image_tt, affine=affine[tt])
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            seg_pred_save_p = os.path.join(args.save_path + 'probability' + '/%s_pred.nii.gz' % (name[tt].split('.')[0]))
            image_save_p = os.path.join(args.save_path + 'image_original' + '/%s.nii.gz'%(name[tt].split('.')[0]))
            nib.save(seg_pred_tt, seg_pred_save_p)
            nib.save(image_tt, image_save_p)
    else:
        for tt in range(image.shape[0]):
            seg_pred0_tt = seg_pred_0[tt]
            seg_pred1_tt = seg_pred_1[tt]
            image_tt = image[tt]
            seg_pred0_tt = nib.Nifti1Image(seg_pred0_tt, affine=affine[tt])
            seg_pred1_tt = nib.Nifti1Image(seg_pred1_tt, affine=affine[tt])
            image_tt = nib.Nifti1Image(image_tt, affine=affine[tt])
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            seg_pred0_save_p = os.path.join(args.save_path + 'organ_probability' + '/%s_organ_pred.nii.gz' % (name[tt].split('.')[0]))
            seg_pred1_save_p = os.path.join(args.save_path + 'tumor_probability' + '/%s_tumor_pred.nii.gz' % (name[tt].split('.')[0]))
            image_save_p = os.path.join(args.save_path + 'image_original' + '/%s.nii.gz'%(name[tt].split('.')[0]))
            nib.save(seg_pred0_tt, seg_pred0_save_p)
            nib.save(seg_pred1_tt, seg_pred1_save_p)
            nib.save(image_tt, image_save_p)
    return None

def validate(args, input_size, model, ValLoader, num_classes, engine):

    for index, batch in enumerate(ValLoader):
        # print('%d processd' % (index))
        image, label, name, task_id, affine = batch
        
        with torch.no_grad():

            pred_sigmoid = predict_sliding(args, model, image.numpy(), input_size, num_classes, task_id)
            # save
            save_nii(args, pred_sigmoid, label, image, name, affine)




def gen_probability():
    """Create the model and start the training."""
    parser = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] ='0'
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)

        cudnn.benchmark = True
        seed = 1234
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create network. 
        model = UNet3D(num_classes=args.num_classes, weight_std=args.weight_std)

        model = nn.DataParallel(model)

        model.eval()

        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)

        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                if args.FP16:
                    checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
                    model.load_state_dict(checkpoint['model'])
                    # optimizer.load_state_dict(checkpoint['optimizer'])
                    # amp.load_state_dict(checkpoint['amp'])
                else:
                    model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))


        valloader, val_sampler = engine.get_test_loader(MOTSValDataSet(args.data_dir, args.val_list))

        print('validate ...')
        validate(args, input_size, [model], valloader, args.num_classes, engine)
        end = timeit.default_timer()
        print(end - start, 'seconds')


