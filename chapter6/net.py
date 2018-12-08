# -*- coding: utf-8 -*-
import torch.nn as nn


def weights_init(m):
    """
    ニューラルネットワークの重みを初期化する。作成したインスタンスに対しapplyメソッドで適用する
    :param m: ニューラルネットワークを構成する層
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:            # 畳み込み層の場合
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:        # 全結合層の場合
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:     # バッチノーマライゼーションの場合
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    """
    生成器Gのクラス 
    """
    def __init__(self, nz=100, nch_g=64, nch=3):
        """
        :param nz: 入力ベクトルzの次元
        :param nch_g: 最終層の入力チャネル数
        :param nch: 出力画像のチャネル数
        """
        super(Generator, self).__init__()
        
        # ニューラルネットワークの構造を定義する
        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.ConvTranspose2d(nz, nch_g * 8, 4, 1, 0),     # 転置畳み込み
                nn.BatchNorm2d(nch_g * 8),                      # バッチノーマライゼーション
                nn.ReLU()                                       # 正規化線形関数
            ),  # (B, nz, 1, 1) -> (B, nch_g*8, 4, 4)
            'layer1': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 8, nch_g * 4, 4, 2, 1),
                nn.BatchNorm2d(nch_g * 4),
                nn.ReLU()
            ),  # (B, nch_g*8, 4, 4) -> (B, nch_g*4, 8, 8)
            'layer2': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 4, nch_g * 2, 4, 2, 1),
                nn.BatchNorm2d(nch_g * 2),
                nn.ReLU()
            ),  # (B, nch_g*4, 8, 8) -> (B, nch_g*2, 16, 16)

            'layer3': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 2, nch_g, 4, 2, 1),
                nn.BatchNorm2d(nch_g),
                nn.ReLU()
            ),  # (B, nch_g*2, 16, 16) -> (B, nch_g, 32, 32)
            'layer4': nn.Sequential(
                nn.ConvTranspose2d(nch_g, nch, 4, 2, 1),
                nn.Tanh()
            )   # (B, nch_g, 32, 32) -> (B, nch, 64, 64)
        })

    def forward(self, z):
        """
        順方向の演算
        :param z: 入力ベクトル
        :return: 生成画像
        """
        for layer in self.layers.values():  # self.layersの各層で演算を行う
            z = layer(z)
        return z


class Discriminator(nn.Module):
    """
    識別器Dのクラス
    """
    def __init__(self, nch=3, nch_d=64):
        """
        :param nch: 入力画像のチャネル数
        :param nch_d: 先頭層の出力チャネル数
        """
        super(Discriminator, self).__init__()

        # ニューラルネットワークの構造を定義する
        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.Conv2d(nch, nch_d, 4, 2, 1),     # 畳み込み
                nn.LeakyReLU(negative_slope=0.2)    # leaky ReLU関数
            ),  # (B, nch, 64, 64) -> (B, nch_d, 32, 32)
            'layer1': nn.Sequential(
                nn.Conv2d(nch_d, nch_d * 2, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 2),
                nn.LeakyReLU(negative_slope=0.2)
            ),  # (B, nch_d, 32, 32) -> (B, nch_d*2, 16, 16)
            'layer2': nn.Sequential(
                nn.Conv2d(nch_d * 2, nch_d * 4, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 4),
                nn.LeakyReLU(negative_slope=0.2)
            ),  # (B, nch_d*2, 16, 16) -> (B, nch_d*4, 8, 8)
            'layer3': nn.Sequential(
                nn.Conv2d(nch_d * 4, nch_d * 8, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 8),
                nn.LeakyReLU(negative_slope=0.2)
            ),  # (B, nch_d*4, 8, 8) -> (B, nch_g*8, 4, 4)
            'layer4': nn.Conv2d(nch_d * 8, 1, 4, 1, 0)
            # (B, nch_d*8, 4, 4) -> (B, 1, 1, 1)
        })

    def forward(self, x):
        """
        順方向の演算
        :param x: 元画像あるいは贋作画像
        :return: 識別信号
        """
        for layer in self.layers.values():  # self.layersの各層で演算を行う
            x = layer(x)
        return x.squeeze()     # Tensorの形状を(B)に変更して戻り値とする

