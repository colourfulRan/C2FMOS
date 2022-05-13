
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
import provider
import numpy as np
import time
from tensorboardX import SummaryWriter
# from thop import profile
import metrics



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
writer = SummaryWriter('run/multi_organ')

classes = ['bg', 'organ', 'tumor']
class2label = {cls: i for i,cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='multi_pointnet2_medicine_tiny', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=128, type=int, help='Epoch to run [default: 128]')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='multi_organ', help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=8192, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int,  default=10, help='Decay step for lr decay [default: every 10 epochs]每10个epoch降低一次学习率')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--iterate', type=int, default=50)

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    train_time = 0
    val_time = 0
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('{}_{}'.format(args.model, args.npoint))
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = './data'
    NUM_CLASSES = 3
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    REMISSION = 3
    ITERATE = args.iterate

    print("start loading training data ...")
    TRAIN_DATASET = MOTSDataset(root=root, sample_points=NUM_POINT, split='train', with_remission=True, iterate=ITERATE, remission=REMISSION)
    print("start loading test data ...")
    TEST_DATASET = MOTSDataset(root=root, sample_points=NUM_POINT, split='test', with_remission=True, iterate=ITERATE, remission=REMISSION)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn = lambda x: np.random.seed(x+int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.label_weights_lut).cuda()

    '''MODEL LOADING'''
    MODEL = importlib.import_module('models.%s' % args.model)
    classifier = MODEL.get_model(NUM_CLASSES, REMISSION, NUM_POINT).cuda()
    # criterion = MODEL.FocalLoss().cuda()
    criterion = MODEL.Focal_Dice_Loss().cuda()
    # criterion = MODEL.DiceLoss().cuda()


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate     # L2范数
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

    global_epoch = 0
    best_iou = 0
    best_dice = 0
    best_acc = 0

    for epoch in range(start_epoch,args.epoch):
        train_start = datetime.datetime.now()
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0


        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, target, _, _, task_ids = data
            points = points.data.numpy()
            points[:,:, :3] = provider.rotate_point_cloud_z(points[:,:, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(),target.long().cuda()
            points = points.transpose(2, 1)
            optimizer.zero_grad()
            classifier = classifier.train()
            seg_pred, _ = classifier(points, task_ids)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, weights)



            loss.backward()
            optimizer.step()
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss

        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))


        cur_train_time = (datetime.datetime.now() - train_start).seconds
        train_time += cur_train_time
        log_string('cur_train_time:%d' % cur_train_time)

        writer.add_scalar('Train/Loss', loss_sum / num_batches, epoch)
        writer.add_scalar('Train/OA', total_correct / float(total_seen), epoch)

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        val_start = datetime.datetime.now()
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

            VOLSMTY = [[] for _ in range(NUM_CLASSES)]
            DICE = [[] for _ in range(NUM_CLASSES)]
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points, target, _, _, task_ids = data
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                classifier = classifier.eval()
                seg_pred, _ = classifier(points,task_ids)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l) )
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) )
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)) )
                    predictions = np.in1d(pred_val.ravel(), l, True).reshape(pred_val.shape).astype(np.uint8)
                    labels = np.in1d(batch_label.ravel(), l, True).reshape(batch_label.shape).astype(np.uint8)
                    confusion_matrix = metrics.ConfusionMatrix(predictions, labels)
                    VOLSMTY[l].append(metrics.VolumeSimilarity(confusion_matrix))
                    DICE[l].append(metrics.DiceCoefficient(confusion_matrix))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            acc_list = np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6)
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(acc_list)))


            cur_val_time = (datetime.datetime.now() - val_start).seconds
            val_time += cur_val_time
            log_string('cur_val_time:%d' % cur_val_time)

            writer.add_scalar('Eval/loss', loss_sum / num_batches, epoch)
            writer.add_scalar('Eval/IoU', mIoU, epoch)
            writer.add_scalar('Eval/OA', total_correct / float(total_seen), epoch)
            writer.add_scalar('Eval/Acc', np.mean(acc_list), epoch)

            iou_per_class_str = '------- IoU && Accuracy--------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f , Acc: %.3f , DICE: %.3f±%.3f , VOLSMTY: %.3f±%.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (12 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]), acc_list[l], float(np.mean(DICE[l])), float(np.std(DICE[l])),
                    float(np.mean(VOLSMTY[l])), float(np.std(VOLSMTY[l])))


            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
            dice = np.mean(DICE[2])
            # dice = (np.mean(DICE[2]) + np.mean(DICE[1])) / 2
            if dice >= best_dice:
                best_dice = dice
                logger.info('Save the best mIoU model...')
                savepath = str(checkpoints_dir) + '/best_dice_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'class_iou': np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6),
                    'class_avg_acc': np.mean(acc_list),
                    'class_acc': acc_list,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')

            log_string('Best dice: %f' % best_dice)
            if np.mean(acc_list) >= best_acc:
                best_acc = np.mean(acc_list)
                logger.info('Save the best mAcc model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'class_iou': np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6),
                    'class_avg_acc': np.mean(acc_list),
                    'class_acc': acc_list,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mAcc: %f' % best_acc)
        global_epoch += 1

    log_string('train_time:%d' % train_time)
    log_string('val_time:%d' % val_time)


if __name__ == '__main__':
    args = parse_args()
    main(args)

