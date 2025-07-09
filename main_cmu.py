import argparse
import torch
from utils.data_loader import CMU_Motion3D
from utils.data_utils import setup_seed
from torch.utils.data import DataLoader
from utils import runtime
from utils.runtime import Trainer
from module.model import GGMNet
from datetime import datetime
import numpy as np
import yaml
import os


def run():
    print('>>> set seed:', args.seed)
    setup_seed(args.seed)
    print('>>> loading dataset_test')
    loaders_test = {}
    len_test = 0
    for act in CMU_Motion3D.define_acts(args.act):
        dataset_test = CMU_Motion3D(args.data, actions=act, input_n=args.past_length, output_n=args.future_length,
                                    split=1, scale=args.scale, test_manner=args.manner, hop=args.n_hop)
        loaders_test[act] = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                       num_workers=0, pin_memory=True)
        len_test += dataset_test.__len__()
    print('>>> Testing dataset length: {:d}'.format(len_test))

    print('>>> create model')
    model = GGMNet(args, dataset_test.group, dataset_test.edges).cuda()
    print(">>> total params: {:.3f} M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    if args.mode == 'train':

        print('>>> loading dataset_train')
        dataset_train = CMU_Motion3D(args.data, actions="all", input_n=args.past_length, output_n=args.future_length,
                                     split=0, scale=args.scale, aug=args.aug, hop=args.n_hop)
        loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=0, pin_memory=True)
        print('>>> Training dataset length: {:d}'.format(dataset_train.__len__()))

        train = Trainer(args, model.parameters(), dataset_test.parent)
        work_dir, log = runtime.exp_create(args.exp, CMU_Motion3D.exp, args.cfg)

        for epoch in range(1, args.epochs + 1):
            lr, pri_loss, aux_loss = train.epoch(model, epoch, loader_train)
            print(f">>> train Epoch: {epoch} Lr: {lr:.7f} Pri_loss: {pri_loss:.7f}, Aux_loss: {aux_loss:.7f}")

            if epoch % args.eval_interval == 0:
                if args.future_length > 12:
                    eval_frame = [(1, 80), (3, 160), (7, 320), (9, 400), (13, 560), (24, 1000)]
                else:
                    eval_frame = [(1, 80), (3, 160), (7, 320), (9, 400)]

                print(f">>> eval Epoch: {epoch} Lr: {lr:.7f} Pri_loss: {pri_loss:.7f}, Aux_loss: {aux_loss:.7f} Manner: {args.manner}", file=log, flush=True)
                avg_mpjpe, avg_avg_mpjpe, eval_msg = runtime.test_cmu(model, eval_frame, CMU_Motion3D.actions,
                                                                      loaders_test, dataset_test.dim_used, args.scale)
                print(eval_msg)
                print(eval_msg, file=log, flush=True)
                state = {'epoch': epoch,
                         'pri_loss': pri_loss,
                         'aux_loss': aux_loss,
                         'state_dict': model.state_dict(),
                         'optimizer': train.optimizer.state_dict()}
                runtime.save_ckpt(work_dir, CMU_Motion3D.exp, state, epoch, avg_avg_mpjpe, args.keep)
    elif args.mode == 'eval':
        print(">>> loading ckpt from '{}'".format(args.ckpt))
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['state_dict'])
        print(">>> ckpt loaded (epoch: {} | pri_loss: {} | aux_loss: {})".
              format(ckpt['epoch'], ckpt['pri_loss'], ckpt['aux_loss']))

        if args.future_length > 12:
            eval_frame = [(1, 80), (3, 160), (7, 320), (9, 400), (13, 560), (24, 1000)]
        else:
            eval_frame = [(1, 80), (3, 160), (7, 320), (9, 400)]

        _, _, eval_msg = runtime.test_cmu(model, eval_frame, CMU_Motion3D.actions, loaders_test, dataset_test.dim_used, args.scale)
        print(eval_msg)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train && eval && infer options')
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    parser.add_argument('--data', type=str, default='./data/cmu', help='path to H36M dataset')
    parser.add_argument('--seed', type=int, default=1234567890, help='random seed')
    parser.add_argument('--cfg', type=str, default='cfg/cmu_short.yml', help='path to the configuration in .yml')
    parser.add_argument('--ckpt', type=str, default='./exp/test.pt', help='path to ckpt')
    parser.add_argument('--exp', type=str, default='./exp', help='dir to release experiment')
    parser.add_argument('--manner', type=str, default='all', help='all or 256 or 8')
    parser.add_argument('--viz', type=str, default='./demo', help='viz save path')
    parser.add_argument('--act', type=str, default='all', help='eval action')
    parser.add_argument('--seq', type=int, default=1, help='from 1 to manner')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        yml_arg = yaml.load(f, Loader=yaml.FullLoader)

    parser.set_defaults(**yml_arg)
    args = parser.parse_args()
    run()




