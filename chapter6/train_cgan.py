# -*- coding:utf-8 -*-
import argparse
import os
import random
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from net import weights_init, Generator, Discriminator


def onehot_encode(label, device, n_class=10):
    """
    カテゴリカル変数のラベルをOne-Hoe形式に変換する
    :param label: 変換対象のラベル
    :param device: 学習に使用するデバイス。CPUあるいはGPU
    :param n_class: ラベルのクラス数
    :return:
    """
    eye = torch.eye(n_class, device=device)
    # ランダムベクトルあるいは画像と連結するために(B, c_class, 1, 1)のTensorにして戻す
    return eye[label].view(-1, n_class, 1, 1)   


def concat_image_label(image, label, device, n_class=10):
    """
    画像とラベルを連結する
    :param image:　画像
    :param label: ラベル
    :param device: 学習に使用するデバイス。CPUあるいはGPU
    :param n_class: ラベルのクラス数
    :return:　画像とラベルをチャネル方向に連結したTensor
    """
    B, C, H, W = image.shape    # 画像Tensorの大きさを取得
    
    oh_label = onehot_encode(label, device)         # ラベルをOne-Hotベクトル化
    oh_label = oh_label.expand(B, n_class, H, W)    # 画像のサイズに合わせるようラベルを拡張する
    return torch.cat((image, oh_label), dim=1)      # 画像とラベルをチャネル方向（dim=1）で連結する


def concat_noise_label(noise, label, device):
    """
    ノイズ（ランダムベクトル）とラベルを連結する
    :param noise: ノイズ
    :param label: ラベル
    :param device: 学習に使用するデバイス。CPUあるいはGPU
    :return:　ノイズとラベルを連結したTensor
    """
    oh_label = onehot_encode(label, device)     # ラベルをOne-Hotベクトル化
    return torch.cat((noise, oh_label), dim=1)  # ノイズとラベルをチャネル方向（dim=1）で連結する


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--nch_g', type=int, default=64)
    parser.add_argument('--nch_d', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--outf', default='./result_cgan', help='folder to output images and model checkpoints')
    
    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    trainset = dset.STL10(root='./dataset/stl10_root', download=True, split='train',
                          transform=transforms.Compose([
                              transforms.RandomResizedCrop(64, scale=(88/96, 1.0), ratio=(1., 1.)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))   # ラベルを使用するのでunlabeledを含めない
    testset = dset.STL10(root='./dataset/stl10_root', download=True, split='test',
                         transform=transforms.Compose([
                             transforms.RandomResizedCrop(64, scale=(88/96, 1.0), ratio=(1., 1.)),
                             transforms.RandomHorizontalFlip(),
                             transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))
    dataset = trainset + testset

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=int(opt.workers))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # 生成器G。ランダムベクトルとラベルを連結したベクトルから贋作画像を生成する
    netG = Generator(nz=opt.nz+10, nch_g=opt.nch_g).to(device)   # 入力ベクトルの次元は、ランダムベクトルの次元nzにクラス数10を加算したもの
    netG.apply(weights_init)
    print(netG)

    # 識別器D。画像とラベルを連結したTensorが、元画像か贋作画像かを識別する
    netD = Discriminator(nch=3+10, nch_d=opt.nch_d).to(device)   # 入力Tensorのチャネル数は、画像のチャネル数3にクラス数10を加算したもの
    netD.apply(weights_init)
    print(netD)

    criterion = nn.MSELoss()

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)

    fixed_noise = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)

    fixed_label = [i for i in range(10)] * (opt.batch_size // 10)  # 確認用のラベル。0〜9のラベルの繰り返し
    fixed_label = torch.tensor(fixed_label, dtype=torch.long, device=device)

    fixed_noise_label = concat_noise_label(fixed_noise, fixed_label, device)  # 確認用のノイズとラベルを連結

    # 学習のループ
    for epoch in range(opt.n_epoch):
        for itr, data in enumerate(dataloader):
            real_image = data[0].to(device)     # 元画像
            real_label = data[1].to(device)     # 元画像に対応するラベル
            real_image_label = concat_image_label(real_image, real_label, device)   # 元画像とラベルを連結

            sample_size = real_image.size(0)
            noise = torch.randn(sample_size, opt.nz, 1, 1, device=device)
            fake_label = torch.randint(10, (sample_size,), dtype=torch.long, device=device)     # 贋作画像生成用のラベル
            fake_noise_label = concat_noise_label(noise, fake_label, device)    # ノイズとラベルを連結
            
            real_target = torch.full((sample_size,), 1., device=device)
            fake_target = torch.full((sample_size,), 0., device=device)

            ############################
            # 識別器Dの更新
            ###########################
            netD.zero_grad()

            output = netD(real_image_label)     # 識別器Dで元画像とラベルの組み合わせに対する識別信号を出力
            errD_real = criterion(output, real_target)
            D_x = output.mean().item()

            fake_image = netG(fake_noise_label)     # 生成器Gでラベルに対応した贋作画像を生成
            fake_image_label = concat_image_label(fake_image, fake_label, device)   # 贋作画像とラベルを連結

            output = netD(fake_image_label.detach())    # 識別器Dで贋作画像とラベルの組み合わせに対する識別信号を出力
            errD_fake = criterion(output, fake_target)
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            ############################
            # 生成器Gの更新
            ###########################
            netG.zero_grad()
            
            output = netD(fake_image_label)     # 更新した識別器Dで改めて贋作画像とラベルの組み合わせに対する識別信号を出力
            errG = criterion(output, real_target)
            errG.backward()
            D_G_z2 = output.mean().item()
            
            optimizerG.step()

            print('[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f} D(x): {:.3f} D(G(z)): {:.3f}/{:.3f}'
                  .format(epoch + 1, opt.n_epoch,
                          itr + 1, len(dataloader),
                          errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if epoch == 0 and itr == 0:
                vutils.save_image(real_image, '{}/real_samples.png'.format(opt.outf),
                                  normalize=True, nrow=10)

        ############################
        # 確認用画像の生成
        ############################
        fake_image = netG(fixed_noise_label)    # 1エポック終了ごとに、指定したラベルに対応する贋作画像を生成する
        vutils.save_image(fake_image.detach(), '{}/fake_samples_epoch_{:03d}.png'.format(opt.outf, epoch + 1),
                          normalize=True, nrow=10)

        ############################
        # モデルの保存
        ############################
        if (epoch + 1) % 50 == 0:
            torch.save(netG.state_dict(), '{}/netG_epoch_{}.pth'.format(opt.outf, epoch + 1))
            torch.save(netD.state_dict(), '{}/netD_epoch_{}.pth'.format(opt.outf, epoch + 1))


if __name__ == '__main__':
    main()
