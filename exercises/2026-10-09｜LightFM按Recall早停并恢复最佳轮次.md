# 2026-10-09｜LightFM按Recall早停并恢复最佳轮次

## 1. 任务背景

LightFM 使用 `fit_partial(..., epochs=1)` 可以逐轮训练，但它不会自动根据验证集指标早停。推荐召回场景通常更关心验证集 `Recall@K`、`NDCG@K` 等指标，而不是训练损失。

今天你将实现一个适用于“指标越大越好”的早停控制器。每轮训练结束后：

1. 计算验证集 `Recall@K`；
2. 判断新指标是否比历史最佳值至少高出 `min_delta`；
3. 有显著提升时保存模型 embedding 和 bias 的独立副本；
4. 连续 `patience` 轮没有显著提升时停止；
5. 最后把模型恢复到最佳轮次，而不是保留停止轮次的参数。

为避免安装 LightFM 影响练习，起始代码用一组验证指标和参数快照模拟逐轮训练结果。你实现的早停逻辑可以直接迁移到真实的 `fit_partial` 循环中。

预计完成时间：20–30 分钟。

## 2. 起始代码

```python
from typing import Dict, List

import numpy as np


# 模拟每轮 fit_partial 后在验证集上得到的 Recall@100。
validation_recalls = np.array(
    [0.120, 0.151, 0.173, 0.174, 0.172, 0.173, 0.171, 0.178],
    dtype=float,
)

PATIENCE = 3
MIN_DELTA = 0.002


def make_fake_lightfm_state(epoch: int) -> Dict[str, np.ndarray]:
    """
    模拟 LightFM 在某一轮结束后的参数。

    真实场景中可保存：
    user_embeddings、item_embeddings、
    user_biases、item_biases。
    """
    return {
        "user_embeddings": np.full((3, 2), epoch, dtype=float),
        "item_embeddings": np.full((4, 2), epoch + 0.1, dtype=float),
        "user_biases": np.full(3, epoch + 0.2, dtype=float),
        "item_biases": np.full(4, epoch + 0.3, dtype=float),
    }


epoch_states = [
    make_fake_lightfm_state(epoch)
    for epoch in range(1, len(validation_recalls) + 1)
]


def validate_inputs(
    metrics: np.ndarray,
    states: List[Dict[str, np.ndarray]],
    patience: int,
    min_delta: float,
) -> None:
    """校验指标、参数快照和早停参数。"""
    # TODO: 完成输入校验；不合法时抛出 ValueError
    raise NotImplementedError


def copy_state(
    state: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    复制一份与原参数完全独立的快照。

    不能只复制字典；字典中的每个 ndarray 也必须复制。
    """
    # TODO: 为每个参数数组创建独立副本
    raise NotImplementedError


def run_early_stopping(
    metrics: np.ndarray,
    states: List[Dict[str, np.ndarray]],
    patience: int,
    min_delta: float,
) -> Dict[str, object]:
    """
    按“指标越大越好”执行早停。

    显著提升定义：
        current_metric > best_metric + min_delta

    返回：
    {
        "best_epoch": 最佳轮次，从 1 开始,
        "best_metric": 最佳验证指标,
        "stop_epoch": 实际停止轮次，从 1 开始,
        "epochs_run": 实际检查的轮数,
        "best_state": 最佳轮次的独立参数副本,
    }
    """
    # TODO: 主动编写逐轮判断循环
    # TODO: 维护 best_metric、best_epoch、bad_epochs 和 best_state
    # TODO: bad_epochs 达到 patience 时立即停止，不再查看后续轮次
    raise NotImplementedError


def restore_state(
    current_state: Dict[str, np.ndarray],
    best_state: Dict[str, np.ndarray],
) -> None:
    """
    将最佳参数原地写回当前模型参数。

    两边键集合和对应数组 shape 必须一致；
    不允许用 current_state = best_state 代替原地恢复。
    """
    # TODO: 校验键和 shape，并使用切片原地复制数组内容
    raise NotImplementedError


result = run_early_stopping(
    metrics=validation_recalls,
    states=epoch_states,
    patience=PATIENCE,
    min_delta=MIN_DELTA,
)

# 模拟停止时模型仍保留最后一轮参数。
current_state = copy_state(
    epoch_states[result["stop_epoch"] - 1]
)
restore_state(current_state, result["best_state"])

print("最佳轮次:", result["best_epoch"])
print("最佳 Recall@100:", result["best_metric"])
print("停止轮次:", result["stop_epoch"])
print("实际运行轮数:", result["epochs_run"])
print(
    "恢复后的 user_embeddings 首个值:",
    current_state["user_embeddings"][0, 0],
)
```

## 3. 学习目标

完成本练习后，你应该能够：

1. 区分“损失越小越好”和“召回指标越大越好”的早停判断方向。
2. 正确使用 `patience` 与 `min_delta` 过滤验证指标的小幅波动。
3. 理解早停时为什么必须保存最佳轮次，而不能只保存最后轮次。
4. 避免浅拷贝导致历史最佳参数被后续训练覆盖。
5. 把独立的早停控制逻辑迁移到 LightFM 的逐轮 `fit_partial` 训练中。

## 4. 待完成任务

1. 实现 `validate_inputs`：
   - `metrics` 必须是一维、非空且全部有限。
   - `states` 长度必须与指标数量一致。
   - 每个参数快照必须是非空字典，值必须为 NumPy 数组。
   - `patience` 必须是正整数，`min_delta` 必须是有限非负数。

2. 实现 `copy_state`：
   - 创建新字典。
   - 对每个参数数组调用复制操作。
   - 修改原始快照时，不得影响复制结果。

3. 实现 `run_early_stopping`：
   - 第 1 轮应直接成为当前最佳轮次。
   - 只有满足 `current > best + min_delta` 才算显著提升。
   - 显著提升时更新最佳指标、轮次和参数副本，并把 `bad_epochs` 清零。
   - 未显著提升时令 `bad_epochs += 1`。
   - `bad_epochs == patience` 时立即停止。
   - 如果直到序列末尾都未触发早停，`stop_epoch` 应为最后一轮。
   - 返回值中的标量使用 Python `int` 或 `float`。

4. 实现 `restore_state`：
   - 键集合不同或对应数组 shape 不同时抛出 `ValueError`。
   - 使用 `current_state[name][...] = best_state[name]` 一类的原地赋值。
   - 保持 `current_state` 原有数组对象不变。

5. 用断言完成自检：
   - 在当前配置下，最佳轮次应为第 3 轮，最佳指标为 `0.173`。
   - 第 4–6 轮都未超过 `0.173 + 0.002`，因此应在第 6 轮停止。
   - `epochs_run` 应为 6，后面的 `0.178` 不应被查看。
   - 恢复后，所有参数内容应与第 3 轮快照一致，而不是第 6 轮。
   - 修改 `epoch_states[2]` 后，`best_state` 仍应保持不变。
   - 将 `patience` 调大后，应能运行到后面的 `0.178`，并选出新的最佳轮次。
