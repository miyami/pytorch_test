# 2026-10-12｜Bootstrap检验召回提升

## 1. 任务背景

昨天你构建了热门召回基线，并计算了用户级 `Recall@K`。现在 LightFM 的平均 Recall 比热门基线高，但均值提升可能来自少量用户，也可能只是有限验证样本带来的波动。

今天使用**配对 Bootstrap**估计平均 Recall 提升的置信区间：

1. 每个用户同时拥有 LightFM 和热门基线两个 Recall；
2. 每次从用户索引中有放回地抽取 N 个用户；
3. 同一组索引同时用于两个模型，保留“同一用户上的配对关系”；
4. 计算本次抽样的平均 Recall 差值；
5. 重复多次，用差值分布的分位数构造置信区间。

如果置信区间整体大于 0，才有较强证据认为 LightFM 优于热门基线；如果区间跨过 0，则当前样本下结论仍不稳定。

预计完成时间：20–30 分钟。

## 2. 起始代码

```python
from typing import Dict

import numpy as np


user_ids = np.array(
    ["u01", "u02", "u03", "u04", "u05",
     "u06", "u07", "u08", "u09", "u10"]
)

# 同一批固定验证用户上的 Recall@100。
popularity_recall = np.array(
    [0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0, 0.5, 0.0, 0.5],
    dtype=float,
)

lightfm_recall = np.array(
    [0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.0],
    dtype=float,
)

N_BOOTSTRAP = 5000
CONFIDENCE_LEVEL = 0.95
SEED = 20261012


def validate_paired_metrics(
    ids: np.ndarray,
    model_metric: np.ndarray,
    baseline_metric: np.ndarray,
) -> None:
    """校验用户 ID 与两组配对指标。"""
    # TODO: 检查维度、长度、唯一性、有限性和 Recall 取值范围
    raise NotImplementedError


def per_user_difference(
    model_metric: np.ndarray,
    baseline_metric: np.ndarray,
) -> np.ndarray:
    """返回每个用户的 model - baseline 指标差。"""
    # TODO: 校验形状一致，并计算逐用户差值
    raise NotImplementedError


def paired_bootstrap_compare(
    ids: np.ndarray,
    model_metric: np.ndarray,
    baseline_metric: np.ndarray,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> Dict[str, object]:
    """
    使用配对用户 Bootstrap 比较两个模型。

    返回：
    {
        "observed_difference": 原样本的平均 Recall 差,
        "ci_low": 置信区间下界,
        "ci_high": 置信区间上界,
        "probability_positive": Bootstrap 差值大于 0 的比例,
        "bootstrap_differences": shape [n_bootstrap] 的差值数组,
    }
    """
    # TODO: 完成参数校验
    # TODO: 使用 np.random.default_rng(seed)
    # TODO: 每次有放回抽取 len(ids) 个用户索引
    # TODO: 同一批索引同时作用于 model 和 baseline
    # TODO: 保存每次抽样的平均差值，并用分位数计算置信区间
    raise NotImplementedError


def interpret_interval(
    ci_low: float,
    ci_high: float,
) -> str:
    """
    返回：
    - "model_better"：区间整体大于 0
    - "baseline_better"：区间整体小于 0
    - "inconclusive"：区间包含 0
    """
    # TODO: 实现三种区间结论
    raise NotImplementedError


result = paired_bootstrap_compare(
    ids=user_ids,
    model_metric=lightfm_recall,
    baseline_metric=popularity_recall,
    n_bootstrap=N_BOOTSTRAP,
    confidence_level=CONFIDENCE_LEVEL,
    seed=SEED,
)

conclusion = interpret_interval(
    result["ci_low"],
    result["ci_high"],
)

print(
    "平均 Recall 提升:",
    round(result["observed_difference"], 4),
)
print(
    "95% 置信区间:",
    (
        round(result["ci_low"], 4),
        round(result["ci_high"], 4),
    ),
)
print(
    "Bootstrap 差值为正的比例:",
    round(result["probability_positive"], 4),
)
print("结论:", conclusion)
```

## 3. 学习目标

完成本练习后，你应该能够：

1. 区分平均指标提升和提升的不确定性。
2. 理解为什么两个模型在同一批用户上比较时应使用配对抽样。
3. 使用可复现的随机数生成器完成有放回 Bootstrap。
4. 用分位数法构造指标差值的置信区间。
5. 根据置信区间是否跨过 0，给出谨慎的模型比较结论。

## 4. 待完成任务

1. 实现 `validate_paired_metrics`：
   - 三个数组都必须是一维、非空且长度一致。
   - 用户 ID 必须唯一。
   - 两组 Recall 必须全部有限，并位于闭区间 `[0, 1]`。

2. 实现 `per_user_difference`：
   - 两组指标必须 shape 完全一致。
   - 返回 `model_metric - baseline_metric`。
   - 不修改任何输入数组。

3. 实现 `paired_bootstrap_compare`：
   - `n_bootstrap` 必须是正整数。
   - `confidence_level` 必须位于开区间 `(0, 1)`。
   - `seed` 必须是整数。
   - 使用 `np.random.default_rng(seed)`，不要修改全局随机状态。
   - 每轮抽取 `len(ids)` 个索引，允许同一用户被重复抽到。
   - 同一轮中，模型和基线必须使用完全相同的用户索引。
   - 观测差值使用原始用户差值的均值。
   - 分位数位置为：
     `alpha / 2` 和 `1 - alpha / 2`，其中 `alpha = 1 - confidence_level`。
   - `probability_positive` 为 Bootstrap 差值严格大于 0 的比例。

4. 实现 `interpret_interval`：
   - `ci_low > 0` 时返回 `model_better`。
   - `ci_high < 0` 时返回 `baseline_better`。
   - 其他情况返回 `inconclusive`。
   - 非有限值或 `ci_low > ci_high` 时抛出 `ValueError`。

5. 用断言完成自检：
   - 当前逐用户差值应为：
     `[0.5, 0.0, 0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.5, -0.5]`。
   - 原样本平均 Recall 提升应为 `0.15`。
   - `bootstrap_differences` 长度应等于 `N_BOOTSTRAP`，并全部有限。
   - 相同 seed 重复运行应得到完全相同的 Bootstrap 数组。
   - 更换 seed 后数组允许变化，但观测差值必须保持不变。
   - 将两个模型输入对调后，观测差值和置信区间应近似取相反数。
   - 对两组完全相同的指标，观测差值和置信区间上下界都应为 `0.0`。
