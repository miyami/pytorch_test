# 2026-11-09｜分组Conformal校准销量区间

> 💬 **[在 ChatGPT 中继续练习](https://chatgpt.com/)**
>
> 当前使用普通 ChatGPT 入口。它不保证自动新建独立会话，也不会自动读取本文件；打开后请新建对话并粘贴下面的启动提示词。
>
> ```text
> 我想完成这篇编程练习：
> https://github.com/miyami/pytorch_test/blob/main/exercises/2026-11-09%EF%BD%9C%E5%88%86%E7%BB%84Conformal%E6%A0%A1%E5%87%86%E9%94%80%E9%87%8F%E5%8C%BA%E9%97%B4.md
>
> 请先读取文件中的第1–4部分，并把它作为本次对话的唯一练习。先用不超过5句话说明任务目标和第一个小步骤，然后等待我提交代码。检查我提交的校准集/测试集隔离、绝对残差、有限样本分位数、分组样本门槛、未知分组回退、区间截断和覆盖率口径；优先指出问题与修改方向，不要直接补齐TODO或提前提供参考答案。只有我明确请求时，才逐级给出分级提示、验收标准、参考答案和复盘问题。
> ```

## 1. 任务背景

销量模型通常只输出一个点预测，但订货决策还需要知道预测的不确定范围。Conformal Prediction 可以使用独立校准集上的预测残差，为测试集构造经验预测区间，而不要求残差服从正态分布。

不同品类的波动程度差异很大：生鲜可能比标品更不稳定。如果所有品类共用同一个区间半径，稳定品类的区间可能过宽，高波动品类又可能覆盖不足。因此今天你将实现“分组校准 + 全局回退”：

- 样本充足的品类使用自己的残差分位数；
- 样本不足或测试集中未见过的品类使用全局半径；
- 销量下界裁剪为 0；
- 在测试集上统计覆盖率、平均区间宽度、低估越界率和高估越界率。

本题只实现 split conformal 的对称绝对残差区间，不训练销量模型。预计完成时间为 20–30 分钟。

## 2. 起始代码

```python
from math import ceil
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# 校准集：点预测来自已经训练完成的模型。
# 不能使用测试集真实销量重新估计区间半径。
calibration = pd.DataFrame(
    {
        "category": (
            ["生鲜"] * 10
            + ["标品"] * 10
            + ["3R"] * 10
        ),
        "y_pred": [
            32, 28, 35, 30, 40, 36, 27, 42, 33, 38,
            18, 20, 22, 19, 24, 21, 23, 17, 25, 20,
            12, 15, 14, 18, 16, 13, 17, 19, 15, 14,
        ],
        "residual": [
            -8, 5, -6, 10, -4, 7, -9, 3, 6, -5,
            -2, 1, 3, -1, 2, -3, 1, 0, -2, 2,
            -4, 3, -2, 5, -3, 2, -5, 4, 1, -2,
        ],
    }
)
calibration["y_true"] = calibration["y_pred"] + calibration["residual"]
calibration = calibration.drop(columns="residual")

test = pd.DataFrame(
    {
        "sample_id": ["T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09"],
        "category": ["生鲜", "生鲜", "标品", "标品", "3R", "3R", "烘焙", "烘焙", "标品"],
        "y_pred": [34, 39, 20, 24, 15, 18, 11, 16, 21],
        "y_true": [29, 49, 18, 27, 10, 20, 15, 13, 20],
    }
)


def validate_frame(
    frame: pd.DataFrame,
    require_target: bool = True,
) -> None:
    """
    校验必需字段、空值、有限值和非负销量。

    require_target=False 时不强制存在 y_true，
    便于未来对尚未产生真实销量的数据构造区间。
    """
    # TODO: 检查category、y_pred以及可选y_true字段
    # TODO: 检查非空、有限值和非负销量
    raise NotImplementedError


def finite_sample_quantile(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """
    计算 split conformal 的有限样本分位数。

    n = len(scores)
    rank = ceil((n + 1) * (1 - alpha))

    rank 从1开始；当rank超过n时截断为n。
    返回排序后第rank个分数，而不是调用默认线性插值分位数。
    """
    # TODO: 校验scores为非空、有限、非负的一维数组
    # TODO: 校验alpha位于(0, 1)
    # TODO: 计算rank并返回对应次序统计量
    raise NotImplementedError


def fit_group_radii(
    calibration: pd.DataFrame,
    group_col: str,
    alpha: float = 0.2,
    min_group_size: int = 5,
) -> Dict[str, object]:
    """
    只使用校准集拟合区间半径。

    返回：
    {
        "global_radius": float,
        "group_radii": {group_value: radius, ...},
        "group_counts": {group_value: count, ...},
    }

    只有样本数 >= min_group_size 的分组进入 group_radii。
    """
    # TODO: 校验输入和min_group_size
    # TODO: 计算abs(y_true - y_pred)
    # TODO: 拟合全局半径
    # TODO: 为样本充足的每个分组拟合独立半径
    raise NotImplementedError


def apply_intervals(
    frame: pd.DataFrame,
    radii: Dict[str, object],
    group_col: str,
    clip_lower_at_zero: bool = True,
) -> pd.DataFrame:
    """
    为点预测增加：
    interval_radius, radius_source, lower, upper

    radius_source 取 "group" 或 "global"。
    未知分组必须回退到 global_radius。
    不允许原地修改输入DataFrame。
    """
    # TODO: 校验radii结构及半径
    # TODO: 逐行选择分组半径或全局半径
    # TODO: 构造上下界，并按参数决定是否裁剪下界
    raise NotImplementedError


def interval_metrics(
    frame_with_intervals: pd.DataFrame,
) -> Dict[str, float]:
    """
    返回：
    coverage：y_true落在闭区间[lower, upper]内的比例；
    average_width：平均区间宽度；
    below_rate：y_true < lower 的比例；
    above_rate：y_true > upper 的比例。
    """
    # TODO: 校验y_true、lower、upper以及上下界顺序
    # TODO: 使用闭区间口径计算四个指标
    raise NotImplementedError


def metrics_by_group(
    frame_with_intervals: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    对每个分组调用 interval_metrics，
    返回每组样本数和四个区间指标。
    """
    # TODO: 分组计算并返回确定顺序的DataFrame
    raise NotImplementedError


radii = fit_group_radii(
    calibration,
    group_col="category",
    alpha=0.2,
    min_group_size=5,
)
test_with_intervals = apply_intervals(
    test,
    radii,
    group_col="category",
    clip_lower_at_zero=True,
)

overall = interval_metrics(test_with_intervals)
by_group = metrics_by_group(test_with_intervals, group_col="category")

print("radii:", radii)
print(test_with_intervals)
print("overall:", overall)
print(by_group)

# 完成后请保留这些最小自检：
assert len(test_with_intervals) == len(test)
assert test_with_intervals["sample_id"].tolist() == test["sample_id"].tolist()
assert test_with_intervals["lower"].ge(0).all()
assert test_with_intervals["upper"].ge(test_with_intervals["lower"]).all()
assert set(test_with_intervals["radius_source"]) <= {"group", "global"}

unseen_rows = test_with_intervals["category"].eq("烘焙")
assert unseen_rows.any()
assert test_with_intervals.loc[unseen_rows, "radius_source"].eq("global").all()
assert test_with_intervals.loc[unseen_rows, "interval_radius"].eq(
    radii["global_radius"]
).all()

assert 0.0 <= overall["coverage"] <= 1.0
assert np.isclose(overall["below_rate"] + overall["coverage"] + overall["above_rate"], 1.0)
```

## 3. 学习目标

完成本练习后，你应能够：

- 使用独立校准集构造 split conformal 预测区间；
- 理解有限样本修正后的分位数秩；
- 区分插值分位数与 conformal 次序统计量；
- 为不同品类拟合差异化残差半径；
- 对小样本和未知分组设计全局回退策略；
- 正确计算覆盖率、区间宽度和两侧越界率；
- 识别使用测试标签调区间所造成的数据泄漏。

## 4. 待完成任务

1. 实现 `validate_frame` 和 `finite_sample_quantile`：
   - 检查字段、形状、有限值和非负销量；
   - 根据给定公式计算 1-based rank；
   - 不使用默认线性插值分位数。

2. 实现 `fit_group_radii`：
   - 用 `abs(y_true - y_pred)` 构造非一致性分数；
   - 始终拟合全局半径；
   - 只为达到样本门槛的分组拟合独立半径；
   - 不读取测试集或修改校准集。

3. 实现 `apply_intervals`：
   - 已知且样本充足的分组使用组内半径；
   - 未知或样本不足的分组使用全局半径；
   - 构造对称区间，并将销量下界裁剪到 0；
   - 保持原始行顺序和索引，不原地修改输入。

4. 实现 `interval_metrics` 和 `metrics_by_group`：
   - 使用闭区间判断覆盖；
   - 分别统计覆盖、下侧越界和上侧越界；
   - 检查三者之和是否为 1；
   - 输出总体与分品类结果。

5. 运行自检，并用三句话解释：
   - 为什么生鲜与标品可能得到不同区间宽度；
   - 为什么测试集中未见过的“烘焙”不能直接拟合自己的半径；
   - 当业务更怕缺货时，对称区间有哪些局限。
