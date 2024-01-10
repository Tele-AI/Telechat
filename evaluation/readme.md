# 榜单复现方法

### CEVAL

首先，下载CEVAL数据集并解压：

```shell
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam.zip
```

之后运行预测脚本：

```python
python score_CEVAL.py --path /path/to/ckpt --five_shot
```

得到提交结果`submission.json`，并提交到[CEVAL官方网站](https://cevalbenchmark.com/)上得到评测结果。


### MMLU

数据下载路径：https://github.com/hendrycks/test?tab=readme-ov-file

之后运行预测脚本：

```python
python score_MMLU.py
```