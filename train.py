import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.models as models
import util


from args_classifier import get_train_args
from collections import OrderedDict
from json import dumps
from models import ClassifierColorCNN, SmallClassifierColorCNN,ClassifierColorUNet
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
import time

from quantization import *


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids  = util.get_available_devices()
    # device = "cuda:1"
    # args.gpu_ids = [1]
    print("gpu Ids: ",args.gpu_ids)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get model
    log.info('Building model...')
    if args.model_type == 'ClassifierColorUNet':
        model = ClassifierColorUNet(313)
    elif args.model_type == 'ClassifierColorCNN':
        model = ClassifierColorCNN(313)
    elif args.model_type == 'SmallClassifierColorCNN':
        model = SmallClassifierColorCNN(313)
    else:
        log.info('Wrong model type.')

    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)
    dtype = torch.float32

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)
    
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, betas=(0.9, 0.99), eps=1e-07,
                           lr=args.lr, weight_decay=args.l2_wd)

    # Get data loader
    log.info('Building dataset...')
    #preprocess = T.Compose([T.Resize(args.input_size),T.RandomHorizontalFlip(),T.RandomAffine(30),T.GaussianBlur((3,3))])
    preprocess = T.Compose([T.Resize(args.input_size)])
    train_dataset = util.TinyImagenetImageFolder(args.train_dir,transform=preprocess)

    # #GET SUBSET
    # train_dataset = torch.utils.data.Subset(train_dataset, np.random.choice(len(train_dataset), 64000, replace=False))

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)
    dev_dataset = util.TinyImagenetImageFolder(args.dev_dir,transform=preprocess)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)

    #Quantizer
    NN = 10
    sigma = 5
    cc = np.load('color_bins.npy')
    cc_pt = torch.tensor(cc)
    encoder = NNEncode(NN,sigma, cc=cc)

    #Class Weights
    if args.prior == True:
        prior_probs = torch.tensor(np.load('prior_probs.npy'))
        gamma = 0.5
        alpha = 1.0
        Prior = PriorFactor(alpha, gamma, prior_probs=prior_probs, device=device)
    else:
        prior_probs = None
        gamma = None
        alpha = None
        Prior = None


    if args.loss_fn == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        log.info('Issue with loss function.')
        return

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for img_gray, img_ab, label in train_loader:
                if len(label) != args.batch_size:
                    continue
                #send to device

                img_gray = img_gray.to(device=device, dtype=dtype)
                t1 = time.time()
                # ab_ground_dist = torch.tensor(encoder.encode_points_mtx_nd(img_ab.numpy())).to(device=device, dtype=dtype)
                ab_ground_dist = encoder.encode_points_mtx_nd_pt(img_ab.to(device),device) # encoder.nbrs_pt.to(device)
                t2 =  time.time()
                # Setup for forward
                batch_size = args.batch_size # fill this in
                optimizer.zero_grad()

                # Forward
                t3 = time.time()
                ab_prob_distribution = model(img_gray)
                t4 = time.time()
                

                # Class weights
                if args.prior == True:
                    weights = Prior.forward(ab_prob_distribution)
                    ab_prob_distribution = ab_prob_distribution * weights
                
                #ab_predicted = encoder.decode_1hot_mtx_nd(ab_prob_distribution.cpu().detach().numpy())


                # TODO: this is ugly rn make it cleaner
                loss = criterion(ab_prob_distribution, ab_ground_dist)

                loss_val = loss.item()

                # Backward
                t5= time.time()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                t6=time.time()
                # print(t2-t1,t4-t3,t6-t5)
                #scheduler.step(step // batch_size)
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         Loss=loss_val)
                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size


                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict, gold_dict = evaluate(model, dev_loader, criterion, device, args.batch_size, encoder)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   gold_dict=gold_dict,
                                   step=step,
                                   split='dev',
                                   num_visuals=args.num_visuals)


def evaluate(model, data_loader, criterion, device, batch_size, encoder):
    loss_meter = util.AverageMeter()
    L2_meter = util.AverageMeter()
    L1_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    gold_dict = {}
    dtype = torch.float32

    l2 = nn.MSELoss()
    l1 = nn.L1Loss()

    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for i,(img_gray, img_ab, label) in enumerate(data_loader):
            if len(label) != batch_size:
                continue
            # Forward
            img_gray = img_gray.to(device=device, dtype=dtype)
            ab_ground_dist = encoder.encode_points_mtx_nd_pt(img_ab.to(device),device)

            # Forward
            ab_prob_distribution = model(img_gray)
            
            ab_predicted = encoder.decode_1hot_mtx_nd_pt(ab_prob_distribution,device)
            pred_dict[i] = ab_predicted
            #ab_predicted = torch.tensor(ab_predicted).to(device=device, dtype=dtype)

            # loss
            loss = criterion(ab_prob_distribution, ab_ground_dist)

            # Other Metrics
            img_ab = img_ab.to(device=device, dtype=dtype)

            l2_loss = l2(ab_predicted, img_ab)
            l1_loss = l1(ab_predicted, img_ab).float()

            L2_meter.update(l2_loss.item(),batch_size)
            L1_meter.update(l1_loss.item(),batch_size)
            loss_meter.update(loss.item(),batch_size)

            # Todo accuracy, precision, recall, f1

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(Loss=loss_meter.avg)

            pred_dict[i] = ab_predicted
            gold_dict[i] = {'answer':img_ab, 'BW':img_gray, 'label': label}


    model.train()

    # how do we want to evaluate this?
    #results = util.eval_dicts(gold_dict, pred_dict)
    results_list = [('CrossEntropyLoss', loss_meter.avg),('L2', L2_meter.avg),('L1', L1_meter.avg)]

    results = OrderedDict(results_list)

    return results, pred_dict, gold_dict


if __name__ == '__main__':
    main(get_train_args())