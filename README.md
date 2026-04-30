# EuroSAT MLP From Scratch

## 目录

- `mlp_eurosat/data.py`: 数据加载、分层划分、标准化
- `mlp_eurosat/model.py`: MLP、激活函数、反向传播、保存/加载权重
- `mlp_eurosat/metrics.py`: Accuracy 和 Confusion Matrix
- `mlp_eurosat/visualize.py`: Loss/Accuracy 曲线、混淆矩阵、第一层权重、错例图
- `train.py`: 训练入口
- `evaluate.py`: 加载最佳权重并测试
- `search.py`: 网格/随机超参数查找

## 环境

可用以下命令导入环境：

`pip install -r requirements.txt`

## 训练

默认使用 `EuroSAT_RGB`，按类别分层划分 train/val/test = 70%/15%/15%，保存验证集 Accuracy 最好的模型。

可用以下命令进行模型训练：

`python train.py --data-dir EuroSAT_RGB --output-dir outputs/run1 --epochs 50 --hidden-dim 512,512 --batch-size 128 --activation relu --lr 0.005 --lr-decay 0.99 --weight-decay 0.0001 --grad-clip 5.0`

如果使用更大的学习率，例如 `--lr 0.05`，请保留默认的 `--grad-clip 5.0`，否则全量数据训练时权重可能快速爆炸，loss 变成 NaN。

训练输出包括：

- `best_model.npz`: 验证集最优权重
- `history.csv`: 每轮 train/val loss、accuracy 和平均梯度范数
- `metrics.json`: 最佳验证结果和最终测试结果
- `training_curves.png`: 训练/验证 Loss 与 Accuracy 曲线
- `confusion_matrix.png`: 测试集混淆矩阵
- `first_layer_filters.png`: 第一层隐藏层权重恢复为图像尺寸后的可视化
- `misclassified_examples.png`: 测试集分类错误样本

## 测试已有权重

可用以下命令测试已有权重：

`python evaluate.py --data-dir EuroSAT_RGB --model outputs/run1/best_model.npz --output-dir outputs/eval_run1`

## 超参数查找

可用以下命令实现超参数查找：

`python search.py --data-dir EuroSAT_RGB --output-dir outputs/search --epochs 50 --hidden-dims 128,256,512 --activations relu,tanh --lrs 0.001,0.005,0.01 --weight-decays 0,0.0001,0.0002`

`search_results.csv` 会记录不同学习率、隐藏层大小、正则强度、激活函数组合下的验证集表现。