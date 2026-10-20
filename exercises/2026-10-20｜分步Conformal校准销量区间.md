# 2026-10-20｜分步Conformal校准销量区间

## 1. 任务背景

昨天你按预测步长评估了 MAE、WAPE 和 MASE，但点预测只能告诉你“最可能是多少”，不能表达预测的不确定性。订货决策还需要知道第 1 天和第 4 天的预测分别有多大波动范围。

今天使用 Split Conformal Prediction 为多步销量预测构造区间：

1. 在独立校准集上计算每个预测步长的绝对残差；
2. 使用有限样本修正后的高分位数作为误差半径；
3. 为测试集点预测生成逐步长上下界；
4. 将销量下界裁剪为 0；
5. 分别计算每个 horizon 的覆盖率和平均区间宽度。

本练习不训练模型，也不复用昨天的 MAE/WAPE/MASE 汇总套路；核心是校准集、有限样本分位数和区间评估。预计完成时间：20–30 分钟。

## 2. 起始代码

```python
from typing import Dict

import math
import torch


# 校准集 shape: [num_calibration_samples=10, horizon=4]
calibration_actual = torch.tensor(
    [
        [12.0, 14.0, 15.0, 17.0],
        [8.0, 9.0, 11.0, 13.0],
        [20.0, 19.0, 18.0, 17.0],
        [5.0, 7.0, 6.0, 8.0],
        [11.0, 12.0, 14.0, 16.0],
        [16.0, 18.0, 19.0, 21.0],
        [7.0, 6.0, 8.0, 9.0],
        [14.0, 15.0, 17.0, 18.0],
        [9.0, 11.0, 10.0, 12.0],
        [18.0, 17.0, 20.0, 22.0],
    ],
    dtype=torch.float32,
)

calibration_prediction = torch.tensor(
    [
        [11.5, 13.0, 13.5, 15.0],
        [8.5, 10.0, 10.0, 11.0],
        [19.0, 18.5, 16.0, 15.0],
        [5.5, 6.0, 7.5, 10.0],
        [10.0, 13.0, 12.5, 14.0],
        [15.5, 16.0, 21.0, 18.0],
        [7.5, 7.0, 6.0, 11.0],
        [13.0, 14.0, 15.0, 20.0],
        [9.5, 10.0, 12.0, 9.0],
        [17.0, 19.0, 17.0, 25.0],
    ],
    dtype=torch.float32,
)

# 测试集 shape: [num_test_samples=5, horizon=4]
test_prediction = torch.tensor(
    [
        [10.0, 12.0, 13.0, 15.0],
        [6.0, 8.0, 9.0, 11.0],
        [15.0, 16.0, 18.0, 20.0],
        [9.0, 10.0, 12.0, 14.0],
        [20.0, 19.0, 21.0, 23.0],
    ],
    dtype=torch.float32,
)

test_actual = torch.tensor(
    [
        [11.0, 13.0, 15.0, 17.0],
        [5.0, 7.0, 8.0, 10.0],
        [14.0, 18.0, 17.0, 23.0],
        [8.0, 11.0, 14.0, 13.0],
        [21.0, 18.0, 24.0, 26.0],
    ],
    dtype=torch.float32,
)

ALPHA = 0.2


def validate_calibration_data(
    actual: torch.Tensor,
    prediction: torch.Tensor,
) -> None:
    """校验二维多步预测数据。"""
    # TODO: 检查维度、shape、dtype、device、有限性和非负性
    raise NotImplementedError


def finite_sample_quantile_level(
    num_calibration_samples: int,
    alpha: float,
) -> float:
    """
    计算有限样本修正后的分位数位置：

        ceil((n + 1) * (1 - alpha)) / n

    最终结果截断到 1.0。
    """
    # TODO: 校验 n 和 alpha
    # TODO: 计算并返回 Python float
    raise NotImplementedError


def conformal_radius_by_horizon(
    calibration_actual: torch.Tensor,
    calibration_prediction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    返回 shape [H] 的各预测步长误差半径。

    nonconformity score = abs(actual - prediction)
    """
    # TODO: 计算绝对残差
    # TODO: 计算有限样本分位数位置
    # TODO: 沿校准样本维使用 higher 插值取分位数
    raise NotImplementedError


def build_prediction_interval(
    prediction: torch.Tensor,
    radius_by_horizon: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    根据 [N, H] 点预测和 [H] 半径构造区间。

    返回：
    {
        "lower": [N, H],
        "upper": [N, H],
    }
    """
    # TODO: 利用广播计算上下界
    # TODO: 将销量下界裁剪到 0
    raise NotImplementedError


def interval_coverage_by_horizon(
    actual: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """返回 shape [H] 的经验覆盖率，区间边界包含等号。"""
    # TODO: 判断每个真实值是否落在闭区间内
    # TODO: 只沿样本维求平均
    raise NotImplementedError


def interval_width_by_horizon(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """返回 shape [H] 的平均区间宽度。"""
    # TODO: 校验上下界并沿样本维求平均
    raise NotImplementedError


validate_calibration_data(
    actual=calibration_actual,
    prediction=calibration_prediction,
)

validate_calibration_data(
    actual=test_actual,
    prediction=test_prediction,
)

radius = conformal_radius_by_horizon(
    calibration_actual=calibration_actual,
    calibration_prediction=calibration_prediction,
    alpha=ALPHA,
)

interval = build_prediction_interval(
    prediction=test_prediction,
    radius_by_horizon=radius,
)

coverage = interval_coverage_by_horizon(
    actual=test_actual,
    lower=interval["lower"],
    upper=interval["upper"],
)

width = interval_width_by_horizon(
    lower=interval["lower"],
    upper=interval["upper"],
)

print("目标覆盖率:", 1.0 - ALPHA)
print("各步长区间半径:", radius)
print("各步长经验覆盖率:", coverage)
print("各步长平均宽度:", width)
```

## 3. 学习目标

完成本练习后，你应该能够：

1. 区分训练集、校准集和测试集在 Conformal Prediction 中的作用。
2. 使用绝对残差构造与模型类型无关的不确定性分数。
3. 理解有限校准样本为什么需要修正分位数位置。
4. 使用广播为多步预测分别构造区间。
5. 同时用覆盖率和区间宽度评价预测区间质量。

## 4. 待完成任务

1. 实现 `validate_calibration_data`：
   - 两个输入都必须是二维浮点 Tensor，布局为 `[N, H]`。
   - shape 必须完全一致，且 `N > 0`、`H > 0`。
   - 两个 Tensor 必须位于同一 device。
   - 所有值必须有限且非负。
   - 不修改输入 Tensor。

2. 实现 `finite_sample_quantile_level`：
   - `num_calibration_samples` 必须是正整数。
   - `alpha` 必须是有限浮点数，位于开区间 `(0, 1)`。
   - 使用
     `ceil((n + 1) * (1 - alpha)) / n`。
   - 当计算结果超过 1 时返回 `1.0`。
   - 返回 Python `float`。

3. 实现 `conformal_radius_by_horizon`：
   - 先调用校验函数。
   - 绝对残差 shape 必须保持为 `[N, H]`。
   - 使用修正后的分位数位置。
   - 调用 `torch.quantile(..., dim=0, interpolation="higher")`。
   - 返回 shape 为 `[H]` 的非负有限 Tensor。
   - 每个 horizon 独立校准，不能把所有残差摊平。

4. 实现 `build_prediction_interval`：
   - `prediction` 必须是二维浮点 Tensor。
   - `radius_by_horizon` 必须是一维，长度等于 `H`。
   - 半径必须全部有限且非负，并与预测位于同一 device。
   - 利用广播计算 `prediction ± radius`。
   - 下界使用 `clamp_min(0.0)`，上界不得裁剪。
   - 保证所有位置满足 `lower <= upper`。

5. 实现 `interval_coverage_by_horizon`：
   - 三个输入必须是相同 shape 的二维 Tensor。
   - 覆盖条件为 `actual >= lower` 且 `actual <= upper`。
   - 将布尔结果转为浮点数，仅沿样本维 `dim=0` 求平均。
   - 输出 shape 必须为 `[H]`，每个值位于 `[0, 1]`。

6. 实现 `interval_width_by_horizon`：
   - `lower` 和 `upper` 必须 shape 相同、全部有限。
   - 每个位置必须满足 `lower <= upper`。
   - 宽度定义为 `upper - lower`。
   - 仅沿样本维求平均，返回 shape `[H]`。

7. 用断言完成自检：
   - 当前修正分位数位置应位于 `(0, 1]`。
   - `radius.shape == coverage.shape == width.shape == (4,)`。
   - 所有半径和宽度都应非负且有限。
   - 下界不得小于 0。
   - 当前数据中较远 horizon 的残差更大，区间半径应体现这种差异。
   - 将测试真实值全部设为点预测时，覆盖率应全部为 1。
   - 增大 `alpha` 后，分位数位置不能增大，区间半径也不应增大。
   - 同时给校准真实值和预测值加相同常数，区间半径应保持不变。
