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


## コードの変更
コードの修正履歴を記載します。<br>現在はColabにPyTorchが初期インストールされているので、Colab利用前のインストールが不要になりました。

- 20190411：7章の物体検出SSDモデルのソースをPyTorch1.0で動くよう修正
- 20200506：7章の物体検出SSDモデルのソースをPyTorch1.5で動くよう修正
- 20230213: 文字列は基本的にシングルクォーテーションを使用するよう統一。一部のインデントが2スペースだったものを4スペースに修正。

## サポート

サンプルプログラムの間違いや動作不具合、書籍中の誤植については本リポジトリのIssuesに投稿ください。

動作不具合についての投稿では、以下を記載ください。

- 実行プログラム名
- エラーメッセージ
- Python、PyTorch、NumPy、Pillow、Matplotlibのバージョン　（必要であればOpenCVのバージョン）
- ハードウェア環境　（GoogleColab or ローカルマシン。　ローカルマシンであれば使用OS。　CPU使用 or GPU使用。）


## 正誤表
| ページ | 誤 | 正 | 補足 |
|:-----------|:------------|:------------|:------------|
| 2章 p29：1個目のプログラムタイトル | 1次元Tensorの例 | numpyへの変換 |  |
| 2章 p36：32ビット、64ビットの浮動小数点dtypeの表記 | torh.float　(cがない) | torch.float |  |
| 2章 p36：8ビット（符号なし、付き）、32ビット（符号付き）、64ビット整数（符号付き）のGPUテンソル | Torch.cudaとtorch.cudaの2つの記載が混在 | torch.cudaに統一 |  |
| 2章 p38：1個目のプログラムタイトル | type()を使った型チェック2 | dtypeを使った型チェック2 |  |
| 2章 p38：1個目のプログラムのメソッド | .type() | .dtype |  |
| 2章 p38：3個目のプログラムのメソッド | .dtype | .type() |  |
| 2章 p39：1個目のプログラムのメソッド | .dtype | .type() |  |
| 2章 p40：形状チェック2の実行結果OutのTensor形状 | 記載漏れ | (2, 3) |  |
| 2章 p46：本文 | torch.tensorのデフォルトではrequires_grad =True | torch.tensorのデフォルトではrequires_grad =False | 入力tensorを作成する多くの場合、requires_gradは指定しません。そのため、入力tensorはデフォルトのrequires_grad = Falseになっています。しかし、torch.nn.Linearやtorch.nn.Conv2dなどが持つ重みWやバイアスbはデフォルトでrequires_grad =Trueになっているため、勾配は計算できます。 |
| 2章 p51: 本文 | ソフトマックス交差エントロピー損失の数式 | 本README下部の記載を参照ください。 | |
| 3章 p65: プログラム | avg_train_loss = train_loss / len(train_loader.dataset) | avg_train_loss = train_loss / len(train_loader) | |
| 3章 p66: プログラム | avg_val_loss = val_loss / len(test_loader.dataset) | avg_val_loss = val_loss / len(test_loader) | |
| 4章 p75：本文 | 以下の例では、nn.Sequentialを使ってクラスを自作していますが、 | 以下の例では、クラスを自作してその内部でnn.Sequentialを使っていますが、 |  |
| 4章 p77: プログラム | avg_train_loss = train_loss / len(train_loader.dataset) | avg_train_loss = train_loss / len(train_loader) | section4_2.ipynb、section4_3.ipynbも同様です。 |
| 4章 p77: プログラム | avg_val_loss = val_loss / len(test_loader.dataset) | avg_val_loss = val_loss / len(test_loader) | section4_2.ipynb、section4_3.ipynbも同様です。 |
| 4章 p80: Pillowのプログラムの変数 | img | image |  |
| 4章 p82: プログラム 1行目のtransforms.Nomalizeの引数 | ([0.485, 0.456, 0.406], [0.229, 0.224,0.225]) | ([0.5, 0.5, 0.5], [0.5, 0.5,0.5]) |  |
| 4章 p82: 本文 | できあががったカスタムデータセットは | できあがったカスタムデータセットは |  |
| 4章 p93: 本文| torchvidsion.models | torchvision.models | |
| 4章 section4_2.ipynb | custom_test_dataset = CustomDataset(root, data_transforms["val"]) | custom_test_dataset = CustomDataset(root, data_transforms["val"], train=False) | 書籍本文では割愛されている、テストデータセットの作成部分です。 |
| 5章 p108: プログラムタイトル | シーケンスが揃っていないデータのLSTM | シーケンスが揃っているデータのLSTM |  |
| 5章 p110: プログラムタイトル | シーケンス長が揃っているデータのLSTM | シーケンス長が揃っているデータのLSTMCell |  |
| 6章 p165: 本文 | 'tain+unlabeled' | 'train+unlabeled' | |
| 6章 p167: 本文 | ずなわち | すなわち | |
| 6章 p172: プログラム| One-Hoe | One-Hot | |
| 6章 p176: 本文 | real_image_lagel | real_image_label | |
| 7章 p182～183: 本文 | mAP:mean average precise | mAP:mean average precision |  |
| 7章 p192: 本文 | 同じクラスのバウンディングボックス通し | 同じクラスのバウンディングボックス同士 | |
| 7章 p209: 本文 | 初期設定の実行回数は120,000回です。 | 初期設定の実行回数は12,000回です。 | ColabのGPU利用時間は12時間ですが、パラメータファイルをargs['save_folder']に保存し、args['resume']にパラメータファイルを指定することで、学習を再開することができます。分けて学習することで、12時間以上学習したパラメータファイルを作成できます。 |
| 7章 p209: プログラム実行 | - | - | 学習実行時に以下のエラーが発生することがあります。これは、Google Driveに格納されたVOCのファイル数が多く、ファイルへのアクセスでタイムアウトが発生していることが原因です。再度実行すると、エラーが解消することがあります。<br>＜エラー＞OSError: [Errno 5] Input/output error: '/content/gdrive/My Drive/Colab Notebooks/pytorch_handbook/chapter7/VOCdevkit/VOC2012/Annotations/2010_003546.xml'<br>＜原因と対応＞https://research.google.com/colaboratory/faq.html#drive-timeout |
| 7章 p224: プログラム（オフセットのネットワークのリスト確認）の中の表記 | Outがない | Outがあり |  |
| 7章 p240: 正解のデータイメージのボックス1の正解ラベル | 19 | 0 |  |
| 8章 p262: 活用メモ | OneHot-Encording | OneHot-Encoding | PyTorch1.系ではtorch.nn.functional.one_hotが使用できます。 |
| 8章 p285: 本文 | - | 実数を四捨五入で整数に丸めます。 | 端数がちょうど0.5のときは結果が偶数になるよう丸められます。 |
| 9章 p297: 活用メモ | Flattend Covolution | Flattened Covolution | |
| 9章 p297: 活用メモ | Flattend Net | Flattened Net | |
| 10章 p365: プログラム | avg_train_loss = train_loss / len(train_loader.dataset) | avg_train_loss = train_loss / len(train_loader) | |
| 10章 p365: プログラム | avg_val_loss = val_loss / len(test_loader.dataset) | avg_val_loss = val_loss / len(test_loader) | |
| 11章 p390: SLT-10のトレーニングデータ、テストデータ、ラベルなしのデータ数 | 5000枚、5000枚、8000枚 | 5000枚、8000枚、100000枚 |  |

**2章 p51 誤**

$$ l(y, t) = -\frac{1}{B}\sum_{i=1}^B \left[ \frac{\sum_{k=0}^{N-1}w^k \cdot t_i^k \cdot \log(\exp(y_i^k))}{\sum_{j=0}^{N-1}\exp(y_i^j)} \right] = -\frac{1}{B}\sum_{i=1}^B \left[ \frac{\sum_{k=0}^{N-1}w^k \cdot t_i^k \cdot y_i^k}{\sum_{j=0}^{N-1}\exp(y_i^j)} \right] $$

**2章 p51 正**

$$ l(y, t) = -\frac{1}{B} \sum_{i=1}^{B} \left[ \sum_{k=0}^{N-1}w^k \cdot t_i^k \cdot \log \left( \frac{\exp(y_i^k)}{\sum_{j=0}^{N-1}\exp(y_i^j)} \right) \right] = -\frac{1}{B}\sum_{i=1}^B \left[ \sum_{k=0}^{N-1} w^k \cdot t_i^k \cdot \left( y_i^k - \log\left(\sum_{j=0}^{N-1}\exp(y_i^j) \right) \right) \right]$$