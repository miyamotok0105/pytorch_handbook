# -*- coding: utf-8 -*-
import torch
import torch.utils.data


def main():
    ## Tensorからのデータセットの作成
    x1 = torch.rand(400, 3)
    y1 = torch.rand(400, 1)
    dataset1 = torch.utils.data.TensorDataset(x1, y1)   # x1とy1からデータセット1を作成
    print('length(dataset1):', len(dataset1))   # データセット1のデータ数
    print('dataset1[0]:', dataset1[0])          # データセット1の最初の行の内容
    print()

    ## データセットの結合
    x2 = torch.rand(600, 3)
    y2 = torch.rand(600, 1)
    dataset2 = torch.utils.data.TensorDataset(x2, y2)   # x2とy2からデータセット2を作成
    print('length(dataset2):', len(dataset2))           # データセット2のデータ数

    dataset3 = torch.utils.data.ConcatDataset([dataset1, dataset2])     # データセット1とデータセット2を結合してデータセット3を作成
    print('length(dataset3):', len(dataset3))                           # データセット3のデータ数
    print()

    ## データセットの分割
    indices4 = [i for i in range(0, 700)]
    indices5 = [i for i in range(700, 1000)]
    dataset4 = torch.utils.data.Subset(dataset3, indices=indices4)  # データセット3の0~699行からデータセット４を作成
    dataset5 = torch.utils.data.Subset(dataset3, indices=indices5)  # データセット3の700~999行からデータセット5を作成
    print('length(dataset4):', len(dataset4))   # データセット4のデータ数
    print('length(dataset5):', len(dataset5))   # データセット5のデータ数
    print('dataset4[0]:', dataset4[0])          # データセット4の最初の行の内容（＝データセット1の最初の行の内容）
    print('dataset5[0]:', dataset5[0])          # データセット5の最初の行の内容
    print()

    ## データセットのランダム分割（重複なし）
    # データセット3からランダムに700行と300行のデータセットを作成
    dataset6, dataset7 = torch.utils.data.random_split(dataset3, [700, 300])
    print('length(dataset6):', len(dataset6))   # データセット6のデータ数
    print('length(dataset7):', len(dataset7))   # データセット7のデータ数
    print('dataset6[0]:', dataset6[0])          # データセット6の最初の行の内容
    print('dataset7[0]:', dataset7[0])          # データセット7の最初の行の内容
    print()

    ## データローダー
    epoch_size = 5      # エポックサイズの設定
    batch_size = 3      # バッチサイズの設定
    num_workers = 2     # データ読み込みに使用するサブプロセス数の設定

    # データセット3を8:2でトレーニングデータセットとバリデーションデータセットに分割
    train_set, valid_set = torch.utils.data.random_split(dataset3, [800, 200])
    print('length(training dataset):', len(train_set))   # トレーニングデータセットのデータ数
    print('length(validation dataset):', len(valid_set))   # バリデーションデータセットのデータ数

    # トレーニングデータセットのデータローダー（シャッフルON）
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # バリデーションデータセットのデータローダー（シャッフルOFF）
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for epoch in range(epoch_size):     # 学習ループ
        print('epoch:', epoch)

        for itr, data in enumerate(train_loader):   # トレーニングのループ
            print('training loop iteration:', itr)
            print(data)     # シャッフルONなのでエポックごとに異なる
            if itr >= 2:
                break
        print()

        for itr, data in enumerate(valid_loader):   # バリデーションのループ
            print('validation loop iteration:', itr)
            print(data)     # シャッフルOFFなのでエポックが変わっても同一
            if itr >= 2:
                break
        print()


if __name__ == '__main__':
    main()