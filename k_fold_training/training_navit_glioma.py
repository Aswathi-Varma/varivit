import os
from functools import partial
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence
from einops import rearrange, repeat
import torch.nn.functional as F
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4,6,7"
import argparse
import json
import pickle
import time
import math
import datetime
import sys
sys.path.append('/root/Projects/varivit')
from resnet_scripts.train_3d_resnet import get_all_feat_and_labels
import numpy as np
import torch
from timm.optim import optim_factory
import utils.lr_decay as lrd
from utils.custom_loss import SoftCrossEntropyWithWeightsLoss
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.backends import cudnn
from utils.used_metrics import roc_auc, get_scores_kfold
from torch.utils.tensorboard import SummaryWriter
from itertools import islice, product
from torch.utils.data import Sampler
import random
import wandb
import pynvml
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import training_script
from dataset.dataset_factory import get_dataset
from environment_setup import PROJECT_ROOT_DIR
from model.model_factory import get_models
from model.model_utils.vit_helpers import interpolate_pos_embed
from read_configs import bootstrap
from utils import misc, lr_sched
import torchio as tio
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset, Dataset
from utils.feature_extraction import generate_features, testing_glioma

class CustomDataset(Dataset):
    def __init__(self, patches, patch_positions, attn_mask, key_pad_mask, batched_image_ids, num_images, batched_labels):
        self.patches = patches
        self.patch_positions = patch_positions
        self.attn_mask = attn_mask
        self.key_pad_mask = key_pad_mask
        self.batched_image_ids = batched_image_ids
        self.num_images = num_images
        self.batched_labels = batched_labels

    def __len__(self):
        return len(self.num_images)

    def __getitem__(self, idx):
        return {
            'patches': self.patches[idx],
            'patch_positions': self.patch_positions[idx],
            'attn_mask': self.attn_mask[idx],
            'key_pad_mask': self.key_pad_mask[idx],
            'batched_image_ids': self.batched_image_ids[idx],
            'num_images': self.num_images[idx],
        }, self.batched_labels[idx]

def custom_collate_fn(batch):
    return {
        'patches': torch.stack([item['patches'] for item in batch]),
        'patch_positions': torch.stack([item['patch_positions'] for item in batch]),
        'attn_mask': torch.stack([item['attn_mask'] for item in batch]),
        'key_pad_mask': torch.stack([item['key_pad_mask'] for item in batch]),
        'num_images': torch.tensor([item['num_images'] for item in batch]),
        'labels': [item['labels'] for item in batch]
    }

def custom_collate(batch):
    """
    Custom collate function that returns a batch of images as a list.
    
    Args:
        batch (list): A list of tuples where each tuple contains the image and its label.

    Returns:
        list: A list containing the images in the batch.
    """
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels

def get_args_parser():
    parser = argparse.ArgumentParser('Glioma training', add_help=False)

    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--volume_size', default=96, type=int,
                        help='images input size')
    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of channels in the input')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR',  # earlier 0
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--dist_on_itp', action='store_true')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Distributed
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--embed_dim', default=1, type=int,
                        help='number of embeddings')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Dataset parameters
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--opt', default=[], type=list,
                        help='dataset options')
    parser.add_argument('--img_sizes', default=[], type=list,
                        help='List of mage sizes')
    parser.add_argument('--use_mixup', action='store_true')

    return parser

def get_dataset_combined(modes, args, transforms):
    dataset = [get_dataset(dataset_name="glioma", mode=mode, args=args, transforms=transforms, use_z_score=args.use_z_score) for mode in modes]
    dataset_combined = ConcatDataset(dataset)
    return dataset_combined

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    return lambda *args: val

def divisible_by(numer, denom):
    return (numer % denom) == 0

def group_images_by_max_seq_len(
    images,
    labels,
    patch_size: int,
    calc_token_dropout = None,
    max_seq_len = 2048

):
    groups = []
    group = []
    seq_len = 0

    for image,label in zip(images, labels):
        assert isinstance(image, torch.Tensor)

        image_dims = image.shape[-3:]
        pl, ph, pw = map(lambda t: t // patch_size, image_dims)

        image_seq_len = (pl * ph * pw)
        image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))

        assert image_seq_len <= max_seq_len, f'image with dimensions {image_dims} exceeds maximum sequence length'

        if (seq_len + image_seq_len) > max_seq_len:
            groups.append(group)
            group = []
            seq_len = 0

        group.append((image, label))
        seq_len += image_seq_len

    if len(group) > 0:
        groups.append(group)

    return groups

def process_data(dataset, device, patch_size=16, channels=4, token_dropout_prob=0.0, group_max_seq_len=1024, group_images=False):
    arange = partial(torch.arange, device=device)
    pad_sequence = partial(orig_pad_sequence, batch_first=True)

    batched_images = []
    batched_labels = []
    p, c = patch_size, channels

    token_dropout_prob = float(token_dropout_prob)
    calc_token_dropout = lambda *args: token_dropout_prob
    has_token_dropout = token_dropout_prob > 0.0

    for image, label in dataset:
        batched_images.append(image.to(device))
        batched_labels.append(label.to(device))

    # Auto pack if specified
    if group_images:
        batched_groups = group_images_by_max_seq_len(
            batched_images, batched_labels,
            patch_size=patch_size,
            calc_token_dropout=calc_token_dropout,
            max_seq_len=group_max_seq_len
        )

    # Process images into variable lengthed sequences with attention mask

    num_images = []
    batched_sequences = []
    batched_positions = []
    batched_image_ids = []
    batched_labels = []
    for groups in batched_groups:
        images = [group[0] for group in groups]
        labels = [group[1] for group in groups]
        num_images.append(len(images))

        sequences = []
        positions = []
        image_ids = torch.empty((0,), device=device, dtype=torch.long)

        for image_id, image in enumerate(images):
            assert image.ndim == 4 and image.shape[0] == c
            image_dims = image.shape[-3:]
            assert all([divisible_by(dim, p) for dim in image_dims]), f'height and width {image_dims} of images must be divisible by patch size {p}'

            pl, ph, pw = map(lambda dim: dim // p, image_dims)

            pos = torch.stack(torch.meshgrid((
                arange(pl),
                arange(ph),
                arange(pw)
            ), indexing='ij'), dim=-1)

            pos = rearrange(pos, 'l h w c -> (l h w) c')
            seq = rearrange(image, 'c (l p1) (h p2) (w p3) -> (l h w) (c p1 p2 p3)', p1=p, p2=p, p3=p)
            seq_len = seq.shape[-2]

            if has_token_dropout:
                token_dropout = calc_token_dropout(*image_dims)
                num_keep = max(1, int(seq_len * (1 - token_dropout)))
                keep_indices = torch.randn((seq_len,), device=device).topk(num_keep, dim=-1).indices

                seq = seq[keep_indices.tolist()]
                pos = pos[keep_indices.tolist()]

            image_ids = F.pad(image_ids, (0, seq.shape[-2]), value=image_id)
            sequences.append(seq)
            positions.append(pos)

        batched_image_ids.append(image_ids)
        batched_sequences.append(torch.cat(sequences, dim=0))
        batched_positions.append(torch.cat(positions, dim=0))
        batched_labels.append(labels)

    # Derive key padding mask

    lengths = torch.tensor([seq.shape[-2] for seq in batched_sequences], device=device, dtype=torch.long)
    max_length = arange(lengths.amax().item())
    key_pad_mask = rearrange(lengths, 'b -> b 1') <= rearrange(max_length, 'n -> 1 n')

    # Derive attention mask, and combine with key padding mask from above

    batched_image_ids = pad_sequence(batched_image_ids)
    attn_mask = rearrange(batched_image_ids, 'b i -> b 1 i 1') == rearrange(batched_image_ids, 'b j -> b 1 1 j')
    attn_mask = attn_mask & rearrange(key_pad_mask, 'b j -> b 1 1 j')

    count_true = torch.sum(attn_mask).item()

    # Combine patched images as well as the patched width/height positions for 3D positional embedding
    patches = pad_sequence(batched_sequences)
    patch_positions = pad_sequence(batched_positions)
    num_images = torch.tensor(num_images, device=device, dtype=torch.long)

    return (patches, patch_positions, attn_mask, key_pad_mask, batched_image_ids, num_images), batched_labels


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
                    max_norm=0, log_writer=None, args=None, mix_up_fn=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # samples = samples.to(device, non_blocking=True)
        targets = torch.cat(targets, dim=0).to(device)

        # if mix_up_fn is not None:
        #     samples, targets = mix_up_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs.to(device), targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, weight):
    # Weights for breast_tumor = 2:1 majority being label 0
    # Since evaluation is always hard target and not SoftTarget
    criterion = torch.nn.CrossEntropyLoss().to(device)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        #images = images.to(device, non_blocking=True)
        target = torch.cat(target, dim=0).to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output.to(device), target)
        outPRED = torch.cat((outPRED, output), 0)
        outGT = torch.cat((outGT, target), 0)

        # batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        # metric_logger.meters['specificity'].update(specificity, n=batch_size)
        # metric_logger.meters['sensitivity'].update(sensitivity, n=batch_size)
    roc_auc_score, specificity, sensitivity, f1_score, probabilities, mcc = get_scores_kfold(predictions=outPRED, target=outGT)
    metric_logger.update(roc_auc_score=roc_auc_score)
    metric_logger.update(specificity=specificity)
    metric_logger.update(sensitivity=sensitivity)
    metric_logger.update(f1_score=f1_score)
    metric_logger.update(mcc=mcc)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* roc_auc_score {:.3f}, f1_score {:.3f}, loss {losses.global_avg:.3f}, mcc {mcc:.3f}'
          .format(roc_auc_score, f1_score, losses=metric_logger.loss, mcc=mcc))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, probabilities, outGT.tolist()

def main(args):

    
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))), force = True)
    print("{}".format(args).replace(', ', ',\n'), force = True)
    
    device = torch.device(args.device)
    print("device ",device, force = True)
    
    args = bootstrap(args=args, key='GLIOMA')

    # Hard-coding the in channels
    args.in_channels = 4
    print("nb classes ", args.nb_classes)
    print("vol size ", args.volume_size)
    print("args.patch_size", args.patch_size)
    args.nb_classes = 2
    args.img_sizes = ["64", "80", "96"]

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    #Wandb initialization
    wandb.init(project="k_fold",
                group="navit",
                config={
                "model" : "navit",
                "dataset" : "glioma",
                "epochs" : args.epochs,
                "warmup_epochs" : args.warmup_epochs,
                "batch_size" : args.batch_size,
                "nb_classes" : args.nb_classes,
                "nb_layers" : 12,
                "nb_heads" : 6,
                "embed_dim" : 384,
                "transforms" : "RandomAffine, RandomNoise, RandomGamma, RandomBlur, RandomFlip",
                },
                sync_tensorboard=True,
                reinit=True)

    cudnn.benchmark = True

    transforms = [
    tio.RandomAffine(),
    tio.RandomNoise(std=0.1),
    tio.RandomGamma(log_gamma=(-0.3, 0.3)),
    tio.RandomBlur(std=(0.1, 2.0)),
    tio.RandomFlip(axes=(0, 1)),
    ]

    train_transforms = tio.Compose(transforms)
    
    # Prepare dataloaders
    modes = [f"{o}_{s}" for o, s in product(["whole"], args.img_sizes)]
    dataset_whole = get_dataset_combined(modes, args, train_transforms)
    dataset_whole_no_aug = get_dataset_combined(modes, args, None)

    #features, labels = get_all_feat_and_labels(dataset_whole, args=args)

    cross_entropy_wt = torch.as_tensor([0.6, 2.5]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=cross_entropy_wt).to(device)

    # Code for the K-fold cross validation
    #kfold_splits = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

    # Create the location for storing split indices
    split_index_path = os.path.join(PROJECT_ROOT_DIR, "glioma", 'k_fold', 'indices_file')
    os.makedirs(split_index_path, exist_ok=True)

    global_probs, global_target, global_roc_auc, global_f1, global_sens, global_spec, global_mcc = [], [], [], [], [], [], []

    #for idx, (train_ids, test_ids) in enumerate(kfold_splits.split(features, labels)):
    for idx in range(5):
        # First we need to save these indices. This would ensure we can reproduce the results
        print(f"Starting for fold {idx}")
        
        if os.path.exists(os.path.join(split_index_path, f"train_{idx}")) and \
                os.path.exists(os.path.join(split_index_path, f"test_{idx}")):
            train_ids = pickle.load(open(os.path.join(split_index_path, f"train_{idx}"), 'rb'))
            test_ids = pickle.load(open(os.path.join(split_index_path, f"test_{idx}"), 'rb'))
            val_ids = pickle.load(open(os.path.join(split_index_path, f"val_{idx}"), 'rb'))
        else:
            print("WARNING: Creating fresh splits.")
            pickle.dump(train_ids, open(os.path.join(split_index_path, f"train_{idx}"), 'wb'))
            pickle.dump(test_ids, open(os.path.join(split_index_path, f"test_{idx}"), 'wb'))
            pickle.dump(val_ids, open(os.path.join(split_index_path, f"val_{idx}"), 'wb'))

        if args.log_dir is not None and not args.eval:
            log_dir = os.path.join(PROJECT_ROOT_DIR, args.log_dir, f'fold_{idx}')
            os.makedirs(log_dir, exist_ok=True)
            log_writer_train = SummaryWriter(log_dir=f"{log_dir}/train_ft")
            log_writer_val = SummaryWriter(log_dir=f"{log_dir}/val_ft")


        #Get dataset_train, dataset_val, dataset_test
        dataset_train = torch.utils.data.Subset(dataset_whole, train_ids)
        dataset_val = torch.utils.data.Subset(dataset_whole_no_aug, val_ids)
        dataset_test = torch.utils.data.Subset(dataset_whole_no_aug, test_ids)

        # Now we process the datasets for navit
        batched_data_train, batched_labels_train = process_data(dataset_train, device, patch_size=args.patch_size, channels=args.in_channels, token_dropout_prob=0.1, group_max_seq_len=1024, group_images=True)
        batched_data_val, batched_labels_val = process_data(dataset_val, device, patch_size=args.patch_size, channels=args.in_channels, token_dropout_prob=0.1, group_max_seq_len=1024, group_images=True)      
        batched_data_test, batched_labels_test = process_data(dataset_test, device, patch_size=args.patch_size, channels=args.in_channels, token_dropout_prob=0.1, group_max_seq_len=1024, group_images=True)

        dataset_train = CustomDataset(patches=batched_data_train[0], patch_positions=batched_data_train[1], attn_mask=batched_data_train[2], key_pad_mask=batched_data_train[3], batched_image_ids=batched_data_train[4], num_images=batched_data_train[5], batched_labels=batched_labels_train)
        dataset_val = CustomDataset(patches=batched_data_val[0], patch_positions=batched_data_val[1], attn_mask=batched_data_val[2], key_pad_mask=batched_data_val[3], batched_image_ids=batched_data_val[4], num_images=batched_data_val[5], batched_labels=batched_labels_val)
        dataset_test = CustomDataset(patches=batched_data_test[0], patch_positions=batched_data_test[1], attn_mask=batched_data_test[2], key_pad_mask=batched_data_test[3], batched_image_ids=batched_data_test[4], num_images=batched_data_test[5], batched_labels=batched_labels_test)

        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, )
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

        model = get_models(model_name='navit', args=args)
        model.to(device)
        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr,)
        print(optimizer)
        loss_scaler = NativeScaler()

        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        output_dir = os.path.join(PROJECT_ROOT_DIR, args.output_dir, f'checkpoints_{idx}')
        os.makedirs(output_dir, exist_ok=True)

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_roc_auc_score, max_spec, max_sen, max_f1, min_loss = 0.0, 0.0, 0.0, 0.0, float('inf')
        for epoch in range(args.start_epoch, args.epochs):
            train_stats = train_one_epoch(
                model, criterion, dataloader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad,
                log_writer=log_writer_train,
                args=args,
                mix_up_fn=None
            )
            # Let us record both train and val stats
            train_val_stats, _, _ = evaluate(dataloader_train, model, device, weight=cross_entropy_wt)
            val_stats, _, _ = evaluate(dataloader_val, model, device, weight=cross_entropy_wt)

            print(f"ROC_AUC score of the network on the val images: {val_stats['roc_auc_score']:.1f}%")
            max_roc_auc_score = select_best_model(args=args, epoch=epoch, loss_scaler=loss_scaler,
                                                max_val=max_roc_auc_score,
                                                model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                                cur_val=val_stats['roc_auc_score'],
                                                model_name=f'best_ft_model_{idx}', output_dir=output_dir)

            max_f1 = select_best_model(args=args, epoch=epoch, loss_scaler=loss_scaler, max_val=max_f1,
                                        model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                        cur_val=val_stats['f1_score'], model_name=f'best_f1_model_{idx}', output_dir=output_dir)
            min_loss = select_min_loss_model(args=args, epoch=epoch, loss_scaler=loss_scaler, min_val=min_loss,
                                        model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                        cur_val=val_stats['loss'], model_name=f'min_loss_model_{idx}', output_dir=output_dir)    


            # Writing the logs
            log_writer_val.add_scalar('ft/roc_auc_score', val_stats['roc_auc_score'], epoch)
            log_writer_val.add_scalar('ft/loss', val_stats['loss'], epoch)
            log_writer_val.add_scalar('ft/f1', val_stats['f1_score'], epoch)
            log_writer_train.add_scalar('ft/roc_auc_score', train_val_stats['roc_auc_score'], epoch)
            log_writer_train.add_scalar('ft/loss', train_val_stats['loss'], epoch)
            log_writer_train.add_scalar('ft/f1', train_val_stats['f1_score'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'train_val_{k}': v for k, v in train_val_stats.items()},
                        **{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if misc.is_main_process():
                log_writer_train.flush()
                log_writer_val.flush()
                with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
        #Delete model
        del model
        torch.cuda.empty_cache()

        model = get_models(model_name='navit', args=args)

        args.finetune = os.path.join(output_dir, f"checkpoint-best_ft_model_{idx}.pth")
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        model.to(device)

        # Evaluate the best model on the test set
        test_stats, probabilities, target  = evaluate(data_loader=dataloader_test, model=model, device=device, weight=None)
        print(f"Accuracy of model on the test images: {test_stats['roc_auc_score']:.1f}%")   
        global_probs.append(probabilities)
        global_target.append(target)
        global_roc_auc.append(test_stats['roc_auc_score'])
        global_f1.append(test_stats['f1_score'])
        global_sens.append(test_stats['sensitivity'])
        global_spec.append(test_stats['specificity'])
        global_mcc.append(test_stats['mcc'])


    plot_ROCAUC(n_folds=5,probabilities=global_probs,labels=global_target, roc_kfold=global_roc_auc)
    print(f"Mean ROC AUC score: {np.mean(np.array(global_roc_auc)):.3f}, Std: {np.std(np.array(global_roc_auc)):.3f}")
    print(f"Mean F1 score: {np.mean(np.array(global_f1)):.3f}, Std: {np.std(np.array(global_f1)):.3f}")
    print(f"Mean Sensitivity: {np.mean(np.array(global_sens)):.3f}, Std: {np.std(np.array(global_sens)):.3f}")
    print(f"Mean Specificity: {np.mean(np.array(global_spec)):.3f}, Std: {np.std(np.array(global_spec)):.3f}")
    print(f"Mean MCC: {np.mean(np.array(global_mcc)):.3f}, Std: {np.std(np.array(global_mcc)):.3f}")

def select_best_model(args, epoch, loss_scaler, max_val, model, model_without_ddp, optimizer, cur_val,
                      model_name='best_ft_model', output_dir=None):
    if cur_val > max_val:
        print(f"saving {model_name} @ epoch {epoch}")
        max_val = cur_val
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=model_name, output_dir=output_dir)  # A little hack for saving model with preferred name
    return max_val

def select_min_loss_model(args, epoch, loss_scaler, min_val, model, model_without_ddp, optimizer, cur_val,
                      model_name='', output_dir=None):
    if cur_val < min_val:
        print(f"saving {model_name} @ epoch {epoch}")
        min_val = cur_val
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=model_name, output_dir=output_dir)  # A little hack for saving model with preferred name
    return min_val 

def plot_ROCAUC(n_folds,probabilities,labels, roc_kfold):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    plt.figure(figsize=(5, 5))
    plt.axes().set_aspect('equal', 'datalim')

    for i in range(n_folds):
        y_score = probabilities[i]
        y_test = labels[i]
        fpr, tpr, _ = roc_curve(y_test, y_score)

        plt.plot(fpr, tpr, 'b', alpha=0.15, label=f'AUC {i + 1} = {roc_kfold[i]:.2f}')
        plt.legend(loc='lower right')
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, 'b', label=f' Mean AUC = {np.mean(np.array(roc_kfold)):.2f}')
    plt.legend(loc='lower right')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3,
                     label=f' Std = {np.std(np.array(roc_kfold)):.2f}')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('roc_auc_ga2.png')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    wandb.finish()