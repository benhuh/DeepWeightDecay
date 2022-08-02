import argparse
def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Meta Learning')
    
    parser.add_argument('-e', '--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-d', '--depth', default=1, type=int,
                        help='depth of linear network (default: 1)')
    # training setting
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_scheduler', default=None, type=str,
                        help='learning rate scheduler',
                        choices=['step', 'multisteps', 'cosine', 'plateau'])
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--clip_gradient', '--cg', default=None, type=float,
                        help='clip the total norm of gradient before update parameter')
    parser.add_argument('--nesterov', action='store_true',
                        help='enable nesterov momentum optimizer')

    # logging
    parser.add_argument('--logdir', default='./log', type=str, help='log path')

    # for testing and validation
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    return parser
