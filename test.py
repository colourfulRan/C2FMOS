import SimpleITK
import nibabel as nib
import argparse
import os
from data_utils.MOTS_dataset import MOTSDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np
import metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


classes = ['bg','organ', 'tumor']
class2label = {cls: i for i,cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def cf_mat(all_result, seg_label_to_cat, pred_val, batch_label):
    for id in seg_label_to_cat.keys():
        idx = np.where(batch_label==id)
        if len(idx) == 0:
            continue
        for i in pred_val[idx]:
            all_result[id][i] += 1
    return all_result

def overall_acc(all_result):
    correct, all_seen = 0, 0
    for id in seg_label_to_cat.keys():
        correct += all_result[id][id]
        all_seen += sum(all_result[id])
    return correct / float(all_seen)

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='multi_pointnet2_medicine_tiny', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--data_dir', type=str, default='/data')
    parser.add_argument("--save_path", type=str, default='/results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='multi_organ', help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=8192, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int,  default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--iterate', type=int, default=100)

    return parser.parse_args()

def main(args):
    def log_string(str):
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)

    experiment_dir = experiment_dir.joinpath('{}_{}'.format(args.model, args.npoint))
    log_dir = experiment_dir.joinpath('logs/')

    '''LOG'''
    args = parse_args()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.data_dir
    NUM_CLASSES = 3
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    REMISSION = 3
    ITERATE = args.iterate

    all_result = {}

    for i in range(len(classes)):
        all_result[i] = [0] * len(classes)
    
    print("start loading test data ...")
    TEST_DATASET = MOTSDataset(root=root, sample_points=NUM_POINT, split='test', with_remission=True, iterate=ITERATE, remission=REMISSION)

    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module('models.%s' % args.model)
    
    classifier = MODEL.get_model(NUM_CLASSES, REMISSION, NUM_POINT).cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_dice_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    log_string('Use pretrain model')


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    epoch = 0


    lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
    if momentum < 0.01:
        momentum = 0.01
    classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))

    '''Evaluate on chopped scenes'''
    with torch.no_grad():
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        VOLSMTY = [[] for _ in range(NUM_CLASSES)]
        DICE = [[] for _ in range(NUM_CLASSES)]
        log_string('---- EVALUATION ----')


        for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points, target, original_xyz, name, task_ids = data
            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            classifier = classifier.eval()
            seg_pred, trans_feat = classifier(points, task_ids)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            batch_label = target.cpu().data.numpy()
            pred_val = np.argmax(pred_val, 2)

            save_nii(args, original_xyz, pred_val, name)
            all_result = cf_mat(all_result, seg_label_to_cat, pred_val, batch_label)
            for l in range(NUM_CLASSES):
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label ==l)))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                predictions = np.in1d(pred_val.ravel(), l, True).reshape(pred_val.shape).astype(np.uint8)
                labels = np.in1d(batch_label.ravel(), l, True).reshape(batch_label.shape).astype(np.uint8)
                confusion_matrix = metrics.ConfusionMatrix(predictions, labels)
                VOLSMTY[l].append(metrics.VolumeSimilarity(confusion_matrix))
                DICE[l].append(metrics.DiceCoefficient(confusion_matrix))

        log_string("\n- \t\t\tbg\t\tspleen\t iou\t Dice\t Volsmty")
        trim = '%s' % args.model


        for i in range(NUM_CLASSES):
            record = all_result[i]
            record = np.array(record) / (sum(record)  + 1e-6) * 100
            print("%s \t %.1f\t %.1f\t %.1f\t %.1f±%.1f\t %.1f±%.1f" % (seg_label_to_cat[i] + ' ' * (8 - len(seg_label_to_cat[i])),
                    record[0],record[1],total_correct_class[i] / float(total_iou_deno_class[i])*100,
                    float(np.mean(DICE[i]))*100, float(np.std(DICE[i]))*100,
                    float(np.mean(VOLSMTY[i]))*100, float(np.std(VOLSMTY[i]))*100))
            trim += ",%.1f" % record[i]

        print(trim)
        print("Overall Accuary : %.1f" % (overall_acc(all_result) * 100))
        print("mIOU : %.1f" % (np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))*100))


def save_nii(args, points, labels, names):
    batch_size, num_point, _ = points.shape
    for i in range(batch_size):
        name = names[i]
        pc_pred = points[i, :]
        name = name[:-9]
        label_name = name+'_label.nii.gz'
        label_path = args.data_dir + os.path.join('/test/label', label_name)
        labelNII = nib.load(label_path)
        label = labelNII.get_data()
        img_shape = label.shape
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        save_pred_path = args.save_path + '/%s_pred.nii.gz' % (name)
        if os.path.exists(save_pred_path):
            img_predNII = nib.load(save_pred_path)
            img_pred = img_predNII.get_data()
        else:
            img_pred = np.zeros((img_shape), dtype=np.uint8)

        for index, (x, y, z) in enumerate(pc_pred):
            img_pred[x, y, z] = labels[i, index]

        # save
        img_pred = nib.Nifti1Image(img_pred, affine=labelNII.affine)

        nib.save(img_pred, save_pred_path)



if __name__ == '__main__':
    args = parse_args()
    main(args)

