# Homework 1 运行说明

## 最终提交文件

直接提交 **`homework-submission.ipynb`**。该文件综合本机 CPU 与 NVIDIA RTX A6000 服务器的实测结果，包含全部 7 题、8 个已执行代码单元、6 张图表、CPU/CUDA 对照表及最终分析；GPU 必需实验已经完成，无需额外补跑。

`homework.ipynb` 和 `homework-cuda.ipynb` 保留为原始实验记录。提交版完整保留服务器的代码及输出，将本机结果写入对照分析，避免混淆设备、checkpoint 或数值来源。下列执行命令用于复现原始实验，不会自动更新提交版；重新运行提交版会用当前环境的结果替换其历史输出。

`homework.ipynb` 完成全部 7 题，包含中文分析、断言自检、训练输出和 6 张图表。运行时保留附带的数据文件与 Notebook 在同一目录。无需使用 `hour-fixed.csv`，原文件未改动。

## 本机复现

已检测到的可用环境为 `/opt/anaconda3/bin/python`（Python 3.12.2）。系统默认的 `python3` 是另一套环境，未安装所需依赖。

从仓库根目录执行：

```bash
/opt/anaconda3/bin/python homework1/run_notebook.py --device cpu
```

执行器使用启动它的 Python 作为 Jupyter kernel，而不是意外使用其他环境。每个单元默认超时 600 秒，失败时返回错误，不覆盖之前成功的 Notebook。成功后保存全部输出和图表，并检查 Notebook 格式、所有单元执行完成及图表数量。

在其他机器上使用 Python 3.12 环境：

```bash
python -m pip install -r homework1/requirements.txt
python homework1/run_notebook.py --device cpu
```

也可在 Jupyter 中选择对应环境，从头依次运行。Notebook 支持从仓库根目录或 `homework1` 目录启动；单独重跑某题前先执行公共初始化单元。

本机验证使用 PyTorch 2.5.1、NumPy 1.26.4、Matplotlib 3.8.4、nbformat 5.9.2、nbclient 0.8.0、ipykernel 6.28.0。不同 PyTorch/硬件版本可能产生小幅数值差异，性能时间也会随机器负载变化。

本次 CPU 完整执行约 16.9 秒，8 个代码单元和全部断言通过，保存 6 个图表。第 6 题的大网络达到约 `7.16e-7` 训练 MSE，但验证 MSE 升至 `6.7814`，已复现明显过拟合。第 7 题最佳验证损失 checkpoint 的准确率为 `59.24%`，多数类基线为 `46.58%`；稀有类别与缺失类别的局限在 Notebook 中说明。

## GPU 服务器复现（专项实验已完成）

以下两项已由上传的服务器记录验证，并纳入最终提交版；这些命令仅用于再次复现。

1. **第 3 题 CUDA 性能实验**：三种矩阵乘法的真实 GPU 计时；已实现预热、前后同步、多次测量和数值一致性检查。
2. **第 4 题 CUDA 显存实验**：比较非原地与 `addmm_` 的当前、峰值、基线外额外峰值和缓存保留显存。本机 CPU 只验证了数值与原地存储行为。

先在服务器安装与 GPU/驱动兼容的 CUDA 版 PyTorch，并确认以下命令输出 `True`：

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -m pip install -r homework1/requirements.txt
python homework1/run_notebook.py --device cuda --output /tmp/homework1-cuda.ipynb
```

CUDA 不可用时会直接报错，不会悄悄回退 CPU。另存输出可保留原来的本机 CPU 记录。完整重跑还会在 GPU 上运行酒质量分类，温度实验仍在 CPU；第 1、2、5、6、7 题没有必须使用 GPU 的任务。

只想补做专项实验时：在服务器 Jupyter 中执行公共初始化单元（默认自动选 CUDA），然后执行第 3、4 题，并保存输出。根据服务器实际结果补充对应分析；当前不包含未经验证的 GPU 数字。

## 实验说明

- 第 3 题默认 `n=128`，`n=5000` 是可选扩展。后者有 2500 万次 Python 循环中的点积调用，可能极慢，不属于必跑任务。
- 第 5 题保留题目原始温度数据（包括 `84.9` 和目标 `4.0`），没有替换成第 6 题的数据；二次项学习率降低以防梯度发散，原训练循环不变。
- 第 6 题比较三组普通网络设置，另有较大网络的过拟合实验；训练/验证数据和预测曲线均保留。
- 第 7 题只用训练集统计量标准化，混淆矩阵对应最低验证损失的 checkpoint；报告多数类基线、macro-F1 和稀有类别表现。随机行划分含重复记录的局限已在分析中说明。
