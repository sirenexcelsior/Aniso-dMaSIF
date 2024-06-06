import os, time, argparse
import os.path as osp
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from pathlib import Path
from progressbar import progressbar
from dataproc import ProteinPairsSurfaces, PairData, CenterPairAtoms
from dataproc import RandomRotationPairAtoms, NormalizeChemFeatures, iface_valid_filter
from anisoconv.transforms import NormalizeArea, NormalizeAxes, GeodesicFPS, SamplePoints
from anisoconv.models import DeltaNetSegmentation
from utils import calc_loss
from iteration import iterate, iterate_surface_precompute

def train(args, writer):
    # 设置随机种子以确保结果可重复
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # 定义数据预处理和变换步骤
    pre_transform = Compose((
        NormalizeArea(),
        NormalizeAxes(),
        SamplePoints(args.num_points * args.sampling_margin, include_normals=True, include_labels=True),
        GeodesicFPS(args.num_points)
    ))

    transform = Compose((
        RandomRotationPairAtoms()
    ))

    # 初始化并加载模型
    net = DeltaNetSegmentation(
        in_channels=3+6,                          
        num_classes=2,                          
        conv_channels=[128]*8,                  
        mlp_depth=1,                            
        embedding_size=512,                     
        num_neighbors=args.k,                   
        grad_regularizer=args.grad_regularizer, 
        grad_kernel_width=args.grad_kernel,     
    ).to(args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)
    best_loss = 1e10  # We save the "best model so far"

    # 从检查点恢复训练（如果有）
    starting_epoch = 0
    if args.restart_training != "":
        checkpoint = torch.load("models/" + args.restart_training)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

    # 加载并预处理训练数据集
    train_dataset = ProteinPairsSurfaces(
        "surface_data", ppi=args.search, train=True, transform=transform, pre_transform=pre_transform
    )
    train_dataset = [data for data in train_dataset if iface_valid_filter(data)]
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True
    )
    print("Preprocessing training dataset")
    train_dataset = iterate_surface_precompute(train_loader, net, args)

    # 分割训练数据和验证数据
    train_nsamples = len(train_dataset)
    val_nsamples = int(train_nsamples * args.validation_fraction)
    train_nsamples = train_nsamples - val_nsamples
    train_dataset, val_dataset = random_split(
        train_dataset, [train_nsamples, val_nsamples]
    )

    # 加载并预处理测试数据集
    test_dataset = ProteinPairsSurfaces(
        "surface_data", ppi=args.search, train=False, transform=transform, pre_transform=pre_transform
    )
    test_dataset = [data for data in test_dataset if iface_valid_filter(data)]
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=True
    )
    print("Preprocessing testing dataset")
    test_dataset = iterate_surface_precompute(test_loader, net, args)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # 训练循环
    for i in range(starting_epoch, args.n_epochs):
        for dataset_type in ["Train", "Validation", "Test"]:
            test = dataset_type != "Train"
            suffix = dataset_type
            dataloader = {
                "Train": train_loader,
                "Validation": val_loader,
                "Test": test_loader
            }[dataset_type]

            info = iterate(
                net,
                dataloader,
                optimizer,
                args,
                test=test,
                summary_writer=writer,
                epoch_number=i,
            )

            for key, val in info.items():
                if key in [
                    "Loss",
                    "ROC-AUC",
                    "Distance/Positives",
                    "Distance/Negatives",
                    "Matching ROC-AUC",
                ]:
                    writer.add_scalar(f"{key}/{suffix}", np.mean(val), i)

                if "R_values/" in key:
                    val = np.array(val)
                    writer.add_scalar(f"{key}/{suffix}", np.mean(val[val > 0]), i)

            if dataset_type == "Validation":  # Store validation loss for saving the model
                val_loss = np.mean(info["Loss"])

        if val_loss < best_loss:
            print("Validation loss {}, saving model".format(val_loss))
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": val_loss,
                },
                model_path + "_epoch{}".format(i),
            )
            best_loss = val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation with dMaSIF Preprocessing')
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size', help='Size of batch (default: 8)')
    parser.add_argument('--n_epochs', type=int, default=50, metavar='num_epochs', help='Number of episodes to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR', help='Learning rate (default: 0.005)')
    parser.add_argument('--k', type=int, default=16, metavar='K', help='Number of nearest neighbors to use (default: 20)')
    parser.add_argument('--grad_kernel', type=float, default=1, metavar='h', help='Kernel size for WLS, as a factor of the average edge length (default: 1)')
    parser.add_argument('--grad_regularizer', type=float, default=0.001, metavar='lambda', help='Regularizer lambda to use for WLS (default: 0.001)')
    parser.add_argument('--sampling_margin', type=int, default=8, metavar='sampling_margin', help='The number of points to sample before using FPS to downsample (default: 8)')
    parser.add_argument('--logdir', type=str, default='', metavar='logdir', help='Root directory of log files. Log is stored in LOGDIR/runs/EXPERIMENT_NAME/TIME. (default: FILE_PATH)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--restart_training', type=str, default='', help='Path to the checkpoint to restart training. The script will continue training from this checkpoint.')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points to sample (default: 1024)')
    parser.add_argument('--search', type=str, default='default_search', help='Search criteria for protein pairs')
    parser.add_argument('--validation_fraction', type=float, default=0.1, help='Fraction of training data to use for validation (default: 0.1)')
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.experiment_name = 'Anisoconv_with_dmasif_preprocessing'
    run_time = time.strftime("%d%b%y_%H_%M", time.localtime(time.time()))

    if args.logdir == '':
        args.logdir = osp.dirname(osp.realpath(__file__))
    args.logdir = osp.join(args.logdir, 'runs', args.experiment_name, run_time)
    writer = SummaryWriter(args.logdir)

    args.checkpoint_dir = osp.join(args.logdir, 'checkpoints')
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    experiment_details = args.experiment_name + '\n--\nSettings:\n--\n'
    for arg in vars(args):
        experiment_details += '{}: {}\n'.format(arg, getattr(args, arg))
    with open(os.path.join(args.logdir, 'settings.txt'), 'w') as f:
        f.write(experiment_details)

    print(experiment_details)
    print('---')
    print('Training...')

    train(args, writer)
