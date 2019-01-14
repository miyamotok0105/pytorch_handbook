# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

import torch.nn as nn
import torch.optim as optim
import torch.utils.data


def main():
    ## 各種設定
    num_workers = 2     # データ読み込みに使用するサブプロセス数の設定
    batch_size = 30     # バッチサイズの設定
    epoch_size = 20     # エポックサイズの設定

    ## データセットとデータローダー
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None,
                     names=['sepal-length',
                            'sepal-width',
                            'petal-length',
                            'petal-width',
                            'class'])   # UCI Machine Learning RepositoryのIrisのデータセットを例として使用

    class_mapping = {label:idx for idx, label in enumerate(np.unique(df['class']))}
    df['class'] = df['class'].map(class_mapping)    # クラスラベルを整数にエンコーディング

    features = torch.tensor(df[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']].values,
                            dtype=torch.float)  # 説明変数のTensor

    labels = torch.tensor(df['class'].values, dtype=torch.long)     # 目的変数のTensor

    dataset = torch.utils.data.TensorDataset(features, labels)  # データセット作成
    # データセットを80:20:50でトレーニングデータセット：バリデーションデータセット：テストデータセットに分割
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths=[80, 20, 50])

    # トレーニングデータセットのデータローダー
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # バリデーションデータセットのデータローダー
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # テストデータセットのデータローダー
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    ## ニューラルネットワークの設定
    net = torch.nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 4)
    )   # MLP
    print(net)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    net.to(device)  # for GPU

    ## 損失関数とオプティマイザーの設定
    criterion = nn.CrossEntropyLoss()                   # 損失関数（ソフトマックス交差エントロピー）
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # オプティマイザー（Adamオプティマイザー）

    ## 学習実行
    epoch_list = []
    train_acc_list = []
    valid_acc_list = []
    for epoch in range(epoch_size):     # エポックのループ

        net.train()  # ニューラルネットを訓練モードに設定
        train_true = []
        train_pred = []
        for itr, data in enumerate(train_loader):   # トレーニングのループ
            features, labels = data
            train_true.extend(labels.tolist())  # クラスラベルのGround-Truthをリストに追加

            features, labels = features.to(device), labels.to(device)   # for GPU

            optimizer.zero_grad()               # 勾配をリセット
            logits = net(features)              # ニューラルネットでロジットを算出
            loss = criterion(logits, labels)    # 損失値を算出
            loss.backward()                     # 逆伝播
            optimizer.step()                    # オプティマイザーでニューラルネットのパラメータを更新

            _, predicted = torch.max(logits.data, 1)    # 最大のロジットからクラスラベルの推論値を算出
            train_pred.extend(predicted.tolist())       # 推論結果をリストに追加

            print('[epochs: {}, mini-batches: {}, records: {}] loss: {:.3f}'.format(
                epoch + 1, itr + 1, (itr + 1) * batch_size, loss.item()))   # 損失値の表示

        net.eval()  # ニューラルネットを評価モードに設定
        valid_true = []
        valid_pred = []
        for itr, data in enumerate(valid_loader):   # バリデーションのループ
            features, labels = data
            valid_true.extend(labels.tolist())  # クラスラベルのGround-Truthをリストに追加

            features, labels = features.to(device), labels.to(device)   # for GPU

            with torch.no_grad():   # バリデーションなので勾配計算OFF
                logits = net(features)

            _, predicted = torch.max(logits.data, 1)    # 最大のロジットからクラスラベルの推論値を算出
            valid_pred.extend(predicted.tolist())       # 推論結果をリストに追加

        train_acc = accuracy_score(train_true, train_pred)      # トレーニングでの正答率をsklearnの機能で算出
        valid_acc = accuracy_score(valid_true, valid_pred)      # バリデーションでの正答率をsklearnの機能で算出

        # エポックごとのトレーニングとバリデーションの正答率を表示
        print('    epocs: {}, train acc.: {:.3f}, valid acc.: {:.3f}'.format(epoch + 1, train_acc, valid_acc))
        print()

        epoch_list.append(epoch + 1)        # ログ用
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

    print('Finished Training')

    print('Save Network')
    torch.save(net.state_dict(), 'model.pth')   # 学習したパラメータを保存

    df = pd.DataFrame({'epoch': epoch_list,
                       'train/accuracy': train_acc_list,
                       'valid/accuracy': valid_acc_list})   # ログ用にデータフレームを作成

    print('Save Training Log')
    df.to_csv('train.log', index=False)     # データフレームをCSVで保存

    ## 学習後の推論実行
    net.eval()  # ニューラルネットを評価モードに設定
    test_true = []
    test_pred = []
    for itr, data in enumerate(test_loader):  # バリデーションのループ
        features, labels = data
        test_true.extend(labels.tolist())  # クラスラベルのGround-Truthをリストに追加

        features, labels = features.to(device), labels.to(device)  # for GPU

        with torch.no_grad():  # バリデーションなので勾配計算OFF
            logits = net(features)

        _, predicted = torch.max(logits.data, 1)  # 最大のロジットからクラスラベルの推論値を算出
        test_pred.extend(predicted.tolist())  # 推論結果をリストに追加

    test_acc = accuracy_score(test_true, test_pred)  # テストでの正答率をsklearnの機能で算出
    print('test acc.: {:.3f}'.format(test_acc))


if __name__ == '__main__':
    main()
