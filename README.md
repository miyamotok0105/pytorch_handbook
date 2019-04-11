# PyTorchニューラルネットワーク実装ハンドブック

- 第1章：PyTorchと開発環境    
- 第2章：PyTorchの基礎    
- 第3章：PyTorchを使ったニューラルネットワーク基礎    
- 第4章：畳み込みニューラルネットワーク    
- 第5章：リカレントニューラルネットワーク    
- 第6章：敵対的生成ネットワーク    
- 第7章：物体検出    
- 第8章〜11章：PyTorchのAPI    


## 付録
紙面に記載できなかった内容を補足しています

- table_dataset.py テーブルデータを処理する例として、UCI Machine Learning RepositoryのIrisデータセットをMLPで学習する
- torch_utils_data.py torch.utils.dataパッケージのいくつかの機能を例示
- torch_utils_data.ipynb torch.utils.dataパッケージのいくつかの機能を例示(jupyternotebook版)

## サポート

返信に数日かかってしまうかもしれませんが、お困りの方は連絡いただければ善処いたします。

miyamotok0105@gmail.com

## 正誤表
| ページ | 誤 | 正 | 補足 |
|:-----------|:------------|:------------|:------------|
| 2章 p29：1個目のプログラムタイトル | 1次元Tensorの例 | numpyへの変換 |  |
| 2章 p36：32ビット、64ビットの浮動小数点dtypeの表記 | torh.float　(cがない) | torch.float |  |
| 2章 p36：8ビット（符号なし、付き）、32ビット（符号付き）、64ビット整数（符号付き）のGPUテンソル | Torch.cudaとtorch.cudaの2つの記載が混在 | torch.cudaに統一 |  |
| 2章 p38：1個目のプログラムタイトル | type()を使った型チェック2 | dtyeを使った型チェック2 |  |
| 2章 p38：1個目のプログラムのメソッド | .type() | .dtype |  |
| 2章 p38：3個目のプログラムのメソッド | .dtype | .type() |  |
| 2章 p39：1個目のプログラムのメソッド | .dtype | .type() |  |
| 2章 p40：形状チェック2の実行結果OutのTensor形状 | 記載漏れ | (2, 3) |  |
| 2章 p46：本文 | torch.tensorのデフォルトではrequires_grad =True | torch.tensorのデフォルトではrequires_grad =False | 入力tensorを作成する多くの場合、requires_gradは指定しません。そのため、入力tensorはデフォルトのrequires_grad = Falseになっています。しかし、torch.nn.Linearやtorch.nn.Conv2dなどが持つ重みWやバイアスbはデフォルトでrequires_grad =Trueになっているため、勾配は計算できます。 |
| 4章 p75：本文 | 以下の例では、nn.Sequentialを使ってクラスを自作していますが、 | 以下の例では、クラスを自作してその内部でnn.Sequentialを使っていますが、 |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
