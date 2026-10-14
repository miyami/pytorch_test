# 2026-10-14｜Newsvendor分位数优化订货量

## 1. 任务背景

你已经能够用平滑后的转化率估计商品期望销量，但实际订货不能只看均值：少订会损失销售和毛利，多订则会产生库存、折价或报损成本。

今天用经典 **Newsvendor（报童模型）**完成单周期订货决策。你将根据缺货成本和过量成本计算临界分位数，再从带权需求场景中确定订货量，并用穷举期望成本验证分位数解。

为了让练习更贴近生鲜短保场景，每个历史需求样本带有权重：近期或相似天气日可以拥有更高权重。最终需要处理权重归一化、离散需求、成本并列和输入异常。

预计完成时间：20–30 分钟。

## 2. 起始代码

```python
from typing import Dict

import numpy as np


# 相似门店在相近天气下的单日需求场景。
demand_scenarios = np.array(
    [8, 10, 11, 12, 12, 13, 14, 15, 17, 20],
    dtype=np.int64,
)

# 越近期、越相似的场景权重越高；输入不保证已经归一化。
scenario_weights = np.array(
    [0.4, 0.6, 0.7, 1.0, 1.2, 1.3, 1.5, 1.4, 1.0, 0.9],
    dtype=float,
)

# 每缺少 1 件的机会损失，以及每多订 1 件的库存/报损成本。
UNDERAGE_COST = 9.0
OVERAGE_COST = 3.0


def validate_inputs(
    demands: np.ndarray,
    weights: np.ndarray,
    underage_cost: float,
    overage_cost: float,
) -> None:
    """校验需求场景、权重和成本参数。"""
    # TODO: 检查维度、长度、数据类型、有限性和取值范围
    raise NotImplementedError


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """返回和为 1 的新权重数组，不修改输入。"""
    # TODO: 校验并归一化权重
    raise NotImplementedError


def critical_fractile(
    underage_cost: float,
    overage_cost: float,
) -> float:
    """计算 Cu / (Cu + Co)。"""
    # TODO: 校验成本并计算临界分位数
    raise NotImplementedError


def weighted_quantile_order(
    demands: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> int:
    """
    返回带权经验分布中第一个累计概率 >= quantile 的需求值。

    注意：
    - demands 可能无序并包含重复值；
    - 相同需求值的权重需要先合并；
    - 返回整数订货量。
    """
    # TODO: 排序需求、合并重复需求的权重
    # TODO: 计算累计概率并定位订货量
    raise NotImplementedError


def expected_cost(
    order_qty: int,
    demands: np.ndarray,
    weights: np.ndarray,
    underage_cost: float,
    overage_cost: float,
) -> float:
    """
    计算给定订货量的带权期望成本。

    shortage = max(demand - order_qty, 0)
    excess = max(order_qty - demand, 0)
    """
    # TODO: 计算每个场景的缺货成本和过量成本
    # TODO: 按归一化权重汇总
    raise NotImplementedError


def exhaustive_optimal_order(
    demands: np.ndarray,
    weights: np.ndarray,
    underage_cost: float,
    overage_cost: float,
) -> Dict[str, object]:
    """
    穷举 0 到 max(demands) 的整数订货量。

    返回：
    {
        "best_order": 最优订货量,
        "best_cost": 最低期望成本,
        "cost_by_order": {订货量: 期望成本},
    }

    如果多个订货量成本相同，选择较小者。
    """
    # TODO: 完成整数候选遍历、成本比较和确定性并列处理
    raise NotImplementedError


validate_inputs(
    demand_scenarios,
    scenario_weights,
    UNDERAGE_COST,
    OVERAGE_COST,
)

normalized_weights = normalize_weights(scenario_weights)

service_quantile = critical_fractile(
    UNDERAGE_COST,
    OVERAGE_COST,
)

quantile_order = weighted_quantile_order(
    demand_scenarios,
    normalized_weights,
    service_quantile,
)

search_result = exhaustive_optimal_order(
    demand_scenarios,
    normalized_weights,
    UNDERAGE_COST,
    OVERAGE_COST,
)

print("临界分位数:", round(service_quantile, 4))
print("分位数订货量:", quantile_order)
print("穷举最优订货量:", search_result["best_order"])
print("最低期望成本:", round(search_result["best_cost"], 4))
```

## 3. 学习目标

完成本练习后，你应该能够：

1. 区分销量均值预测与成本敏感的订货决策。
2. 根据缺货成本和过量成本计算 Newsvendor 临界分位数。
3. 从无序、带重复值的带权需求场景中计算经验分位数。
4. 将缺货损失与库存损失统一为可比较的期望成本。
5. 用离散穷举验证分位数解，并处理并列最优的确定性规则。

## 4. 待完成任务

1. 实现 `validate_inputs`：
   - `demands` 和 `weights` 必须是一维、非空且长度一致。
   - 需求必须是非负整数。
   - 权重必须全部有限且非负，并且权重总和大于 0。
   - 两类成本必须是有限正数。

2. 实现 `normalize_weights`：
   - 返回浮点数组，且元素之和应接近 `1.0`。
   - 不修改原始权重数组。
   - 对非法权重抛出 `ValueError`。

3. 实现 `critical_fractile`：
   - 使用公式 `underage_cost / (underage_cost + overage_cost)`。
   - 返回值必须位于开区间 `(0, 1)`。

4. 实现 `weighted_quantile_order`：
   - `quantile` 必须位于闭区间 `[0, 1]`。
   - 先按需求升序排列，再合并重复需求值的权重。
   - 使用归一化后的累计权重。
   - 返回第一个累计概率大于或等于目标分位数的需求值。
   - 边界分位数 `0` 和 `1` 分别返回最小、最大需求值。

5. 实现 `expected_cost`：
   - `order_qty` 必须是非负整数。
   - 用向量化操作分别计算缺货量和过量量。
   - 返回按归一化权重汇总的浮点期望成本。
   - 不修改任何输入数组。

6. 实现 `exhaustive_optimal_order`：
   - 枚举 `0, 1, ..., max(demands)`。
   - 为每个候选量调用 `expected_cost`。
   - 以不超过 `1e-12` 的差异视为成本并列，并选择较小订货量。
   - 返回完整的 `cost_by_order` 字典，方便检查成本曲线。

7. 用断言完成自检：
   - 归一化权重之和应接近 `1.0`，原数组保持不变。
   - 当前成本下临界分位数应为 `0.75`。
   - 分位数订货量应与穷举最优订货量一致。
   - 最优成本应不高于相邻可行订货量的成本。
   - 提高缺货成本后，最优订货量不能下降。
   - 提高过量成本后，最优订货量不能上升。
   - 将所有权重同时乘以同一正数，不应改变最优订货量。
