# 基于cnn_glove的文本分类算法

## 简介
数据集和GLOVE词向量下载请询问：1041589975@qq.com


## 运行方法

### 训练
run `python train.py` to train the cnn with the <strong>spam and ham files (only support chinese!)</strong> (change the config filepath in FLAGS to your own)

### 在tensorboard上查看summaries
run `tensorboard --logdir /{PATH_TO_CODE}/runs/{TIME_DIR}/summaries/` to view summaries in web view

### 测试、分类
run `python eval.py --checkpoint_dir /{PATH_TO_CODE/runs/{TIME_DIR}/checkpoints}`<br/>
如果需要分类自己提供的文件，请更改相关输入参数

    如果需要测试准确率，需要指定对应的标签文件(input_label_file):
    python eval.py --input_label_file /PATH_TO_INPUT_LABEL_FILE
    说明：input_label_file中的每一行是0或1，需要与input_text_file中的每一行对应。
    在eval.py中，如果有这个对照标签文件input_label_file，则会输出预测的准确率

### 推荐运行环境
python 3.x
tensorflow 1.0.0  



