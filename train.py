import numpy as np
import torch
import torch.nn as nn
from torch import optim
import argparse
import time
import re
import os
from model import *
from loss import *
from datasets.dataset import Data
import cfg
import log
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from PIL import Image
from sklearn.model_selection import KFold

from eval_code.epm2labelmap import epm2labelmap
from eval_code.eval_shape_detection import shape_detection
from eval_code.voi import voi
from eval_code.rand import adapted_rand
from eval_code.pixel_accuracy import pixel

import pdb

def train(args):
    data_root = cfg.config_all[args.dataset]['data_root']
    data_lst = cfg.config_all[args.dataset]['data_lst']
    mean_bgr = np.array(cfg.config_all[args.dataset]['mean_bgr'])
    nm = np.loadtxt(os.path.join(data_root, data_lst), dtype=str)
    crop_size = args.crop_size
    logger = args.logger

    # Split three folds
    kf = KFold(n_splits=3)
    kf.get_n_splits(nm)


    best_coco_list_cv = []
    voi_score_all_fold = []
    rand_score_all_fold = []
    accuracy_all_fold = []
    best_weight_fold = []

    save_weight_path_alpha = os.path.join(args.param_dir + args.dataset , str(args.alpha) + '_' + str(args.base_lr))
    if not os.path.exists(save_weight_path_alpha):
        os.makedirs(save_weight_path_alpha)

    res_dir_alpha = os.path.join(args.res_dir + args.dataset, str(args.alpha) + '_' + str(args.base_lr))
    if not os.path.exists(res_dir_alpha):
        os.makedirs(res_dir_alpha)

    for index_cv, (train_index, val_index) in enumerate(kf.split(nm)):
        model = UNet(n_channels=args.channels, n_classes=args.classes)

        if args.cross_validation_pretrain_dir + args.dataset:
            cv_pretrain_path = os.path.join(args.cross_validation_pretrain_dir + args.dataset, 'unet_best_cross_val_'+str(index_cv)+'.pth')
            if os.path.exists(cv_pretrain_path):
                pretrain_weights = torch.load(cv_pretrain_path)
                model.load_state_dict(pretrain_weights)
                logger.info('-- Load cross validation pretrain weight {}'.format(cv_pretrain_path))
            else:
                print('Model is not found')
                break

        else:
            def weights_init(m):
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                    torch.nn.init.zeros_(m.bias)

            # KAIMING initialization
            model.apply(weights_init)
            logger.info('Kaiming initliazation done.')

        # Change it to adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)

        train_name_lst = nm[train_index]
        val_name_lst = nm[val_index]
        train_img = Data(data_root, train_name_lst, mean_bgr=mean_bgr, crop_size=crop_size)
        trainloader = torch.utils.data.DataLoader(train_img, batch_size=args.batch_size, shuffle=True, num_workers=0)
        n_train = len(trainloader)

        val_img = Data(data_root, val_name_lst, mean_bgr=mean_bgr, crop_size=crop_size)
        valloader = torch.utils.data.DataLoader(val_img, batch_size=args.batch_size, shuffle=False, num_workers=0)

        start_time = time.time()
        if args.cuda:
            model.cuda()

        if args.resume:
            logger.info('resume from %s' % args.resume)
            state = torch.load(args.resume)
            optimizer.load_state_dict(state['solver'])
            model.load_state_dict(state['param'])

        # Writer will output to ./runs/ directory by default
        writer = SummaryWriter()
        epochs = args.epochs
        max_rand_score_epoch = 0
        voi_score_all_epoch = []
        rand_score_all_epoch = []
        accuracy_all_epoch = []

        for epoch in range(epochs):
            model.train()
            mean_loss = []
            mean_unet_loss = []
            mean_distance_loss = []
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', leave=False) as pbar:
                for images, contours, seeds, boundaries in trainloader:
                    optimizer.zero_grad()

                    if args.cuda:
                        images, contours, seeds, boundaries = images.cuda(),  contours.cuda(), seeds.cuda(), boundaries.cuda()

                    out = model(images)
                
                    # cv2.imwrite('abc.png', (torch.sigmoid(out).squeeze().cpu().data.numpy()*255).astype(np.uint8))
                    # pdb.set_trace()

                    # 1. calculate the BCE loss
                    # 2. Compare the distance loss function
                    unet_loss = cross_entropy_loss2d(out, contours,  args.cuda, args.balance)
                    distance_loss = args.alpha * DAHU_loss(out, seeds, boundaries, contours)
                    loss = unet_loss + distance_loss 

                    loss.backward()
                    optimizer.step()

                    # Update the pbar
                    pbar.update(images.shape[0])
                    mean_loss.append(loss)
                    mean_unet_loss.append(unet_loss)
                    mean_distance_loss.append(distance_loss)

                    pbar.set_postfix(**{'Distance loss': distance_loss.item(), 'unet_loss': unet_loss.item() })

            save_weight_path = os.path.join(save_weight_path_alpha, 'param_cross_validation_' + str(index_cv))
            if not os.path.exists(save_weight_path):
                os.makedirs(save_weight_path)

            # Save UNET weights
            if (epoch+1) % args.snapshots == 0:
                torch.save(model.state_dict(), '%s/unet_%d.pth' % (save_weight_path, epoch+1))
                state = {'step': epoch+1, 'param':model.state_dict(),'solver':optimizer.state_dict()}
                torch.save(state, '%s/unet_%d.pth.tar' % (save_weight_path, epoch+1))

            tm = time.time() - start_time

            save_dir = os.path.join(res_dir_alpha, 'cross_validation_' + str(index_cv), 'unet' + '_' + str(epoch) + '_fuse')

            # Use evaluation model to test samples
            model.eval()
            val_mean_loss = []
            for i, (val_images, val_labels, val_seeds, val_boundaries) in enumerate(valloader):
                if args.cuda:
                    val_images, val_labels, val_seeds, val_boundaries = val_images.cuda(),  val_labels.cuda(), val_seeds.cuda(), val_boundaries.cuda()

                with torch.no_grad():
                    out = model(val_images)

                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                fuse = torch.sigmoid(out).cpu().numpy()[0, 0, :, :]
                fuse = Image.fromarray(fuse)
                fuse.save(os.path.join(save_dir, '{}.tif'.format(val_name_lst[i][0].split('/')[1].split('.')[0])))

                val_total_loss = 0
                val_unet_loss    =  cross_entropy_loss2d(out, val_labels,  args.cuda, args.balance)
                val_distance_map =  args.alpha * DAHU_loss(out, val_seeds, val_boundaries, val_labels)
                val_total_loss   =  val_distance_map + val_unet_loss
                val_mean_loss.append(val_total_loss)

                # Add scalar to tensorboard Loss/Train
                writer.add_scalars('Loss/train/val', {
                                    'Train loss': torch.mean(torch.stack(mean_loss)),
                                    'Validation loss': torch.mean(torch.stack(val_mean_loss))
                }, epoch)

            logger.info('lr: %e, loss: %f, unet_loss: %f, distance_loss: %f, validation loss: %f, time using: %f' %
                 (optimizer.param_groups[0]['lr'],
                 torch.mean(torch.stack(mean_loss)), torch.mean(torch.stack(mean_unet_loss)), torch.mean(torch.stack(mean_distance_loss)),
                 torch.mean(torch.stack(val_mean_loss)), tm))

            start_time = time.time()
            
            # Evaluation and early stop code
            gt_dir = os.path.join(res_dir_alpha, 'cross_validation_' +  str(index_cv), str(epoch) + '_gt_labeled_map')
            if not os.path.exists(gt_dir):
                os.makedirs(gt_dir)
            
            # Create Evaluation results folder
            eval_save_path = os.path.join(res_dir_alpha, 'cross_validation_' + str(index_cv), str(epoch) +  '_evaluation')
            if not os.path.exists(eval_save_path):
                os.makedirs(eval_save_path)

            voi_val_s_image = []
            voi_val_m_image = []
            rand_val_image = []
            accuracy_image = []

            output_dir = eval_save_path
            for t in os.listdir(save_dir):
                input_path = os.path.join(save_dir, t)
                output_path = os.path.join(output_dir, t)

                # Input gt path
                input_gt_path = os.path.join(args.label_dir, t.split('.')[0]+'.png')
                gt_path = os.path.join(gt_dir, t)
                input_contenders_path = [output_path]

                # Transfer label map to gt label map
                epm2labelmap(input_gt_path, gt_path, args.EPM_threshold, debug_labels=None)

                # Transfer epm to label map
                epm2labelmap(input_path, output_path, args.EPM_threshold, debug_labels=None)

                # Calculate Pixel

                accuracy = pixel(input_path, input_gt_path)
                accuracy_image.append(accuracy)

                # Calculate VOI score
                hxgy, hyxg = voi(input_contenders_path[0], gt_path)
                voi_val_s_image.append(hxgy)
                voi_val_m_image.append(hyxg)

                # Calculate ARI score
                rand_score = adapted_rand(input_contenders_path[0], gt_path)
                rand_val_image.append(1-rand_score)

            
            # Measure the current
            current = np.mean(rand_val_image)
            if current >= max_rand_score_epoch:
                max_rand_score_epoch = current
                current_best_weight = epoch
                torch.save(model.state_dict(), '{}/unet_dist_loss_{}_{}.pth'.format(save_weight_path_alpha, 'best_cross_val', str(index_cv))) # Save best weight
            else:
                pass

            voi_score_all_epoch.append(np.mean(voi_val_s_image)+np.mean(voi_val_m_image))
            rand_score_all_epoch.append(np.mean(rand_val_image))
            accuracy_all_epoch.append(np.mean(accuracy_image))

            # Index of max coco value
            index_max_rand = np.argmax(rand_score_all_epoch)

            print('Average adapted rand index:           {}'.format(round(np.mean(rand_val_image),    4)))
            print('Average split error in VOI:           {}'.format(round(np.mean(voi_val_s_image),      4)))
            print('Average merge error in VOI:           {}'.format(round(np.mean(voi_val_m_image),      4)))
            print('Average error in VOI:                 {}'.format(round(np.mean(voi_val_s_image)+np.mean(voi_val_m_image),      4)))
            print('Average error in accuracy:                 {}'.format(round(np.mean(accuracy_image),      4)))

            print('Current best average rand_score results: {}'.format(round(rand_score_all_epoch[index_max_rand],     4)))
            print('Current best weight:                  {}'.format(round(current_best_weight,     4)))          
            print('Current best average voi_score results: {}'.format(round(voi_score_all_epoch[index_max_rand],     4)))
            print('Current best average accuracy results: {}'.format(round(accuracy_all_epoch[index_max_rand],     4)))


            if epoch - current_best_weight == args.Early_stop_limit:
                print('Reach early stop limit, stop the training.')
                best_coco_list_cv.append(max_rand_score_epoch)
                break
            else:
                pass

            # Learning rate schedular to change learning
            scheduler.step(current)
            print('Current learning rate                 {}'.format(optimizer.param_groups[0]['lr']))

            start_time = time.time()

        index_max_rand = np.argmax(rand_score_all_epoch)
        voi_score_all_fold.append(voi_score_all_epoch[index_max_rand])
        rand_score_all_fold.append(rand_score_all_epoch[index_max_rand])
        accuracy_all_fold.append(accuracy_all_epoch[index_max_rand])
        best_weight_fold.append(current_best_weight)

    print('***************************************************************')
    print('Average variation of information(VOI)                 {} (±{})'.format(np.round(np.mean(voi_score_all_fold), 4), np.round(np.std(voi_score_all_fold), 4)))
    print('Average ARI                 {} (±{})'.format(np.round(np.mean(rand_score_all_fold), 4), np.round(np.std(rand_score_all_fold), 4)))
    print('Average accuracy                 {} (±{})'.format(np.round(np.mean(accuracy_all_fold), 4), np.round(np.std(accuracy_all_fold), 4)))
    print('Best weights fold : ', best_weight_fold)
    print('Best ARI: ', rand_score_all_fold)
    print('***************************************************************')
    
    with open(args.res_dir + args.dataset + '/summary.txt', 'w') as f:
        f.write("Average adapted rand index(ARI): %s , %s\n" % (str(np.round(np.mean(rand_score_all_fold), 4)), str(np.round(np.std(rand_score_all_fold), 4)) ))
        f.write("Average variation of information(VOI): %s , %s\n" % (str(np.round(np.mean(voi_score_all_fold), 4)), str(np.round(np.std(voi_score_all_fold), 4)) ))
        f.write("Average accuracy: %s , %s\n" % (str(np.round(np.mean(accuracy_all_fold), 4)), str(np.round(np.std(accuracy_all_fold), 4)) ))

        for item in best_weight_fold:
            f.write("%s\n" % item)

def main():
    args = parse_args()

    # Choose the GPUs
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger = log.get_logger(args.log)
    args.logger = logger
    logger.info('*'*80)
    logger.info('the args are the below')
    logger.info('*'*80)
    for x in args.__dict__:
        logger.info(x+','+str(args.__dict__[x]))
    logger.info(cfg.config[args.dataset])
    logger.info('*'*80)

    if not os.path.exists(args.param_dir + args.dataset):
        os.mkdir(args.param_dir + args.dataset)

    torch.manual_seed(args.seed)

    train(args)

def parse_args():
    AUC_THRESHOLD_DEFAULT = 0.5

    parser = argparse.ArgumentParser(description='Train UNET for different args')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
        default='CREMI', help='The dataset to train')
    parser.add_argument('--seed', type=int, default=100,
        help='Seed control.')
    parser.add_argument('--param_dir', type=str, default='params-shuffle',
        help='the directory to store the params')
    parser.add_argument('--lr', dest='base_lr', type=float, default=1e-3,
        help='the base learning rate of model')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='the momentum')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('--weight-decay', type=float, default=0.0002,
        help='the weight_decay of net')
    parser.add_argument('-r', '--resume', type=str, default=None,
        help='whether resume from some, default is None')
    parser.add_argument('--model', type=str, default=None,
        help='Pre-load model')
    parser.add_argument('--epochs', type=int, default=200,
        help='Epoch to train network, default is 100')
    parser.add_argument('--max-iter', type=int, default=40000,
        help='max iters to train network, default is 40000')
    parser.add_argument('--iter-size', type=int, default=10,
        help='iter size equal to the batch size, default 10')
    parser.add_argument('--average-loss', type=int, default=50,
        help='smoothed loss, default is 50')
    parser.add_argument('-s', '--snapshots', type=int, default=1,
        help='how many iters to store the params, default is 1000')
    parser.add_argument('--step-size', type=int, default=50,
        help='the number of iters to decrease the learning rate, default is 50')
    parser.add_argument('-b', '--balance', type=float, default=1.1,
        help='the parameter to balance the neg and pos, default is 1.1')
    parser.add_argument('-l', '--log', type=str, default='log.txt',
        help='the file to store log, default is log.txt')
    parser.add_argument('-k', type=int, default=1,
        help='the k-th split set of multicue')
    parser.add_argument('--batch-size', type=int, default=1,
        help='batch size of one iteration, default 1')
    parser.add_argument('--crop-size', type=int, default=None,
        help='the size of image to crop, default not crop')
    parser.add_argument('--complete-pretrain', type=str, default=None,
        help='finetune on the complete_pretrain, default None')
    parser.add_argument('--side-weight', type=float, default=0.5,
        help='the loss weight of sideout, default 0.5')
    parser.add_argument('--fuse-weight', type=float, default=1.1,
        help='the loss weight of fuse, default 1.1')
    parser.add_argument('--gamma', type=float, default=0.1,
        help='the decay of learning rate, default 0.1')
    parser.add_argument('--channels', type=int, default=1,
        help='number of channels for unet')
    parser.add_argument('--classes', type=int, default=1,
        help='number of classes in the output')
    parser.add_argument('--label_dir', type=str, default='/lrde/image/CV_2021_yizi/CREMI/contour',
        help='the dir to label ground truth')
    parser.add_argument('--res_dir', type=str, default='training_info_suffle',
        help='the dir to store result')
    parser.add_argument('--auc-threshold', type=float,
        help='Threshold value (float) for AUC: 0.5 <= t < 1.'f' Default={AUC_THRESHOLD_DEFAULT}', default=AUC_THRESHOLD_DEFAULT)
    parser.add_argument('--EPM_threshold', type=int, default=0.5,
        help='Threshold to create binary image of EPM')
    parser.add_argument('--alpha', type=float, default=0.1,
        help='alpha value for weight balancing.')
    parser.add_argument('--Early_stop_limit', type=int, default=20,
        help='Early stop limit.')
    parser.add_argument('--cross_validation_pretrain_dir', type=str, default='params-shuffle',
        help='The cross validation pretrain weight directory.')

    return parser.parse_args()

if __name__ == '__main__':
    main()
