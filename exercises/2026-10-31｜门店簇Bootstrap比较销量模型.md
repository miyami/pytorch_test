# 2026-10-31｜门店簇Bootstrap比较销量模型

> 💬 **[在 ChatGPT 中继续练习](https://chatgpt.com/)**
>
> 当前使用普通 ChatGPT 入口。它不保证自动新建独立会话，也不会自动读取本文件；打开后请新建对话并粘贴下面的启动提示词。
>
> ```text
> 我想完成这篇编程练习：
> https://github.com/miyami/pytorch_test/blob/main/exercises/2026-10-31%EF%BD%9C%E9%97%A8%E5%BA%97%E7%B0%87Bootstrap%E6%AF%94%E8%BE%83%E9%94%80%E9%87%8F%E6%A8%A1%E5%9E%8B.md
>
> 请先读取文件中的第1–4部分，并把它作为本次对话的唯一练习。先用不超过5句话说明任务目标和第一个小步骤，然后等待我提交代码。检查我提交的语法、统计逻辑、边界条件和数据泄漏；优先指出问题与修改方向，不要直接补齐TODO或提前提供参考答案。只有我明确请求时，才逐级给出提示、验收标准、参考答案和复盘问题。
> ```

## 1. 任务背景

新销量模型的整体 MAE 比基线低，并不代表改进足够稳定。同一门店内的多天销量通常相关，如果把所有“门店×日期”行独立重采样，会破坏这种相关结构，并可能低估结果的不确定性。

今天你将实现一个成对的门店簇 Bootstrap：

1. 同时保留每行的真实销量、基线预测和候选模型预测；
2. 每次以门店为单位进行有放回抽样；
3. 被抽中的门店保留其全部日期记录；
4. 计算每次抽样中的成对 MAE 差值；
5. 给出差值的百分位置信区间和候选模型改进概率；
6. 查看各门店差值，识别“整体变好但局部门店退化”的情况。

约定差值为：

[
Delta MAE = MAE_{candidate} - MAE_{baseline}
]

因此，(Delta MAE < 0) 表示候选模型更好。预计完成时间为 20–30 分钟。

## 2. 起始代码

```python
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


evaluation = pd.DataFrame(
    [
        ["S1", "2026-10-01", 20.0, 18.0, 20.0],
        ["S1", "2026-10-02", 22.0, 25.0, 23.0],
        ["S1", "2026-10-03", 19.0, 17.0, 20.0],
        ["S1", "2026-10-04", 25.0, 28.0, 24.0],
        ["S2", "2026-10-01", 40.0, 36.0, 39.0],
        ["S2", "2026-10-02", 42.0, 47.0, 43.0],
        ["S2", "2026-10-03", 38.0, 35.0, 39.0],
        ["S2", "2026-10-04", 45.0, 49.0, 46.0],
        ["S3", "2026-10-01", 12.0, 11.0, 13.0],
        ["S3", "2026-10-02", 15.0, 14.0, 15.0],
        ["S3", "2026-10-03", 14.0, 16.0, 15.0],
        ["S3", "2026-10-04", 13.0, 12.0, 14.0],
        ["S4", "2026-10-01", 70.0, 65.0, 68.0],
        ["S4", "2026-10-02", 75.0, 82.0, 78.0],
        ["S4", "2026-10-03", 68.0, 62.0, 67.0],
        ["S4", "2026-10-04", 80.0, 87.0, 82.0],
        ["S5", "2026-10-01", 30.0, 31.0, 33.0],
        ["S5", "2026-10-02", 28.0, 29.0, 31.0],
        ["S5", "2026-10-03", 33.0, 32.0, 36.0],
        ["S5", "2026-10-04", 35.0, 34.0, 38.0],
    ],
    columns=[
        "store_id",
        "date",
        "actual",
        "baseline_pred",
        "candidate_pred",
    ],
)
evaluation["date"] = pd.to_datetime(evaluation["date"])


@dataclass
class BootstrapSummary:
    baseline_mae: float
    candidate_mae: float
    observed_delta: float
    ci_low: float
    ci_high: float
    improvement_probability: float


def validate_evaluation_data(data: pd.DataFrame) -> None:
    """校验评估表结构、主键、日期类型和数值合法性。"""
    # TODO: 校验必需列、非空、(store_id, date) 唯一
    # TODO: 校验 date 类型与三个数值列的有限性、非负性
    # TODO: 确保至少包含两个门店，且每个门店至少有两行
    # TODO: 不得原地修改输入表
    raise NotImplementedError


def mean_absolute_error(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """计算 MAE，并对维度、长度、空数组和有限性进行校验。"""
    # TODO: 校验两个一维数组
    # TODO: 计算并返回 Python float
    raise NotImplementedError


def paired_mae_delta(data: pd.DataFrame) -> float:
    """
    计算 candidate MAE - baseline MAE。

    必须在同一批样本行上分别计算两个模型的 MAE。
    """
    # TODO: 使用 mean_absolute_error 计算两个 MAE
    # TODO: 返回候选模型减基线模型的差值
    raise NotImplementedError


def store_level_deltas(data: pd.DataFrame) -> pd.Series:
    """
    计算每个门店的成对 MAE 差值。

    返回以 store_id 为索引、按 store_id 升序排列的浮点 Series。
    """
    # TODO: 按门店分组
    # TODO: 对每个完整门店组调用 paired_mae_delta
    raise NotImplementedError


def cluster_bootstrap_deltas(
    data: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """
    以门店为簇执行有放回 Bootstrap。

    每次抽取的门店数量等于原始唯一门店数。
    同一门店如果被抽中多次，其全部日期记录也应重复进入样本。
    返回 shape=[n_bootstrap] 的 float64 数组。
    """
    # TODO: 校验 n_bootstrap 和 seed
    # TODO: 创建局部 np.random.default_rng(seed)
    # TODO: 预先保存每个门店的完整行组
    # TODO: 每轮有放回抽取门店，拼接完整门店组
    # TODO: 计算该轮 paired_mae_delta
    raise NotImplementedError


def percentile_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """使用双侧百分位数法计算置信区间。"""
    # TODO: 校验一维、非空、有限，以及 0 < confidence < 1
    # TODO: 计算对称的上下分位点
    raise NotImplementedError


def summarize_bootstrap(
    data: pd.DataFrame,
    bootstrap_deltas: np.ndarray,
    confidence: float = 0.95,
) -> BootstrapSummary:
    """
    汇总原始样本指标与 Bootstrap 结果。

    improvement_probability 定义为 bootstrap_deltas < 0 的比例；
    恰好等于 0 不计为改进。
    """
    # TODO: 计算原始 baseline MAE、candidate MAE 和 observed delta
    # TODO: 计算置信区间与改进概率
    # TODO: 返回 BootstrapSummary
    raise NotImplementedError


validate_evaluation_data(evaluation)

per_store = store_level_deltas(evaluation)

bootstrap_deltas = cluster_bootstrap_deltas(
    data=evaluation,
    n_bootstrap=2000,
    seed=42,
)

summary = summarize_bootstrap(
    data=evaluation,
    bootstrap_deltas=bootstrap_deltas,
    confidence=0.95,
)

print("各门店 ΔMAE:")
print(per_store)
print("\nBootstrap 汇总:")
print(summary)
```

## 3. 学习目标

完成本练习后，你应该能够：

1. 使用成对指标差值比较两个销量模型。
2. 理解为什么同一门店内样本相关时不应逐行独立重采样。
3. 实现以门店为单位的有放回簇 Bootstrap。
4. 正确处理同一门店在一次抽样中被抽中多次的情况。
5. 使用百分位数法构造模型差值的置信区间。
6. 用改进概率表达候选模型优于基线的稳定程度。
7. 结合门店级差值识别整体指标掩盖的局部退化。

## 4. 待完成任务

1. 实现 `validate_evaluation_data`：
   - 必须包含题目给出的五列，不得有缺失值。
   - `(store_id, date)` 必须唯一。
   - `date` 必须是 datetime 类型。
   - `actual`、`baseline_pred`、`candidate_pred` 必须有限且非负。
   - 至少包含两个门店，每个门店至少有两条记录。
   - 函数不得改变输入表的内容、索引或行顺序。

2. 实现 `mean_absolute_error`：
   - 两个输入必须是一维、等长、非空数组。
   - 所有元素必须有限。
   - 返回 Python `float`。
   - 不允许调用 sklearn 的指标函数。

3. 实现 `paired_mae_delta`：
   - 基线和候选模型必须使用完全相同的真实值与样本行。
   - 差值方向固定为“候选减基线”。
   - 不得先按门店算 MAE 后再对门店做简单平均。

4. 实现 `store_level_deltas`：
   - 按完整门店数据分别计算差值。
   - 返回结果按 `store_id` 升序排列。
   - 本题中应能看到至少一个门店的差值大于 0，说明候选模型在该门店退化。

5. 实现 `cluster_bootstrap_deltas`：
   - 使用局部随机数生成器，不要调用全局 `np.random.seed`。
   - 每轮抽取的门店数等于原始门店数，且必须有放回。
   - 被抽中的门店要保留全部日期，不能从门店内再抽行。
   - 门店被抽中两次时，其全部行必须在该轮样本中出现两次。
   - 每轮计算成对 `ΔMAE`，最终返回 `float64` 数组。
   - 相同 seed 和输入必须得到完全相同的结果。
   - 关键重采样循环由学习者完成。

6. 实现 `percentile_interval`：
   - 置信水平必须严格位于 0 和 1 之间。
   - 95% 区间使用 2.5% 和 97.5% 分位点。
   - 返回两个有限的 Python `float`，且 `low <= high`。

7. 实现 `summarize_bootstrap`：
   - 原始指标只能使用未重采样的完整数据计算。
   - `observed_delta` 应等于 `candidate_mae - baseline_mae`。
   - 改进概率为 Bootstrap 差值严格小于 0 的比例。
   - 所有字段都返回 Python `float`。

8. 完成自检：
   - `bootstrap_deltas.shape == (2000,)`，dtype 为 `float64`。
   - 同一 seed 连续运行两次结果完全一致。
   - 更换 seed 后结果通常不同，但原始三个 MAE 字段不变。
   - 手工构造“两个模型预测完全相同”的数据，验证差值、区间和改进概率均为 0。
   - 比较“逐行 Bootstrap”和“门店簇 Bootstrap”的区间宽度，并解释为何两者可能不同。
