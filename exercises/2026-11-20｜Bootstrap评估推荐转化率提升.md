# 2026-11-20｜Bootstrap评估推荐转化率提升

> 预计用时：20–30 分钟

## 在 ChatGPT 中继续练习

[打开 ChatGPT](https://chatgpt.com/)

这是普通入口，不保证自动创建独立会话，也不会自动加载本文件。请在新对话中复制下面的启动提示词：

> 请先读取这篇练习的第 1–4 部分：https://github.com/miyami/pytorch_test/blob/main/exercises/2026-11-20%EF%BD%9CBootstrap%E8%AF%84%E4%BC%B0%E6%8E%A8%E8%8D%90%E8%BD%AC%E5%8C%96%E7%8E%87%E6%8F%90%E5%8D%87.md。我会随后提交代码。请先检查我的独立重采样、分块实现、绝对提升、百分位置信区间和随机种子复现，优先指出最小可修复问题；只有在我请求时才逐级给出提示。不要直接补全 TODO，不要提前提供参考答案或第 7 部分，并在代码可运行后逐项验收。

## 1. 任务背景

推荐策略上线前，实验组转化率高于对照组并不代表策略一定有效：有限样本中的随机波动也可能产生表面提升。你需要使用非参数 Bootstrap，估计“实验组转化率 − 对照组转化率”的抽样分布。

本练习以用户为独立观测单位，每位用户只有一个二元转化结果。每次 Bootstrap：

1. 在对照组用户内有放回抽样，样本量保持不变；
2. 在实验组用户内独立有放回抽样，样本量保持不变；
3. 计算两组转化率之差；
4. 重复多次后，计算标准误、百分位置信区间和提升为正的概率。

工程上不能一次创建无限大的采样矩阵。本题要求使用 `chunk_size` 分块生成重采样结果，同时保证：

- 相同随机种子产生完全一致的分布；
- 最后一块不足 `chunk_size` 时仍准确生成指定次数；
- 输入 DataFrame 不被修改；
- 组内用户 ID 唯一，转化值只能为 0 或 1；
- 相对提升只作为描述性结果，置信区间针对绝对提升。

## 2. 起始代码

```python
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapResult:
    control_rate: float
    treatment_rate: float
    absolute_uplift: float
    relative_uplift: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    probability_positive: float
    n_bootstrap: int
    seed: int


def make_ab_data(
    n_control: int = 600,
    n_treatment: int = 640,
    seed: int = 42,
) -> pd.DataFrame:
    """生成用户级二元转化数据。"""
    rng = np.random.default_rng(seed)

    control = pd.DataFrame(
        {
            "user_id": [f"c_{i:04d}" for i in range(n_control)],
            "group": "control",
            "converted": rng.binomial(1, 0.185, size=n_control),
        }
    )
    treatment = pd.DataFrame(
        {
            "user_id": [f"t_{i:04d}" for i in range(n_treatment)],
            "group": "treatment",
            "converted": rng.binomial(1, 0.218, size=n_treatment),
        }
    )
    return pd.concat([control, treatment], ignore_index=True)


def validate_ab_data(data: pd.DataFrame) -> None:
    """检查列、组别、用户唯一性、缺失值和二元转化标签。"""
    # TODO: 实现输入校验
    raise NotImplementedError


def summarize_conversion(data: pd.DataFrame) -> pd.DataFrame:
    """
    返回列：
    group、users、conversions、conversion_rate。

    组别顺序固定为 control、treatment。
    """
    # TODO: 按组汇总人数、转化人数和转化率
    raise NotImplementedError


def bootstrap_rate_differences(
    control: np.ndarray,
    treatment: np.ndarray,
    n_bootstrap: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    分块生成实验组转化率减对照组转化率。

    返回形状为 [n_bootstrap] 的 float 数组。
    """
    # TODO: 校验参数
    # TODO: 每块分别对两组做有放回抽样并计算均值差
    # TODO: 正确处理最后一个不完整块
    raise NotImplementedError


def evaluate_uplift(
    data: pd.DataFrame,
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    chunk_size: int = 500,
    seed: int = 2026,
) -> tuple[BootstrapResult, np.ndarray]:
    """
    返回统计结果与完整 Bootstrap 绝对提升分布。

    百分位区间使用两侧等尾分位数。
    standard_error 使用样本标准差（ddof=1）。
    probability_positive 为分布中差值 > 0 的比例。
    """
    # TODO: 校验输入和 ci_level
    # TODO: 提取两组数组并计算观测转化率
    # TODO: 调用分块 Bootstrap，计算标准误、区间和正提升概率
    raise NotImplementedError


data = make_ab_data()
data_before = data.copy(deep=True)

summary = summarize_conversion(data)
result, uplift_distribution = evaluate_uplift(
    data,
    n_bootstrap=5000,
    ci_level=0.95,
    chunk_size=384,
    seed=2026,
)

print(summary)
print(result)


# 输入不可被修改
pd.testing.assert_frame_equal(data, data_before)


# 汇总结构与基础关系
assert list(summary.columns) == [
    "group",
    "users",
    "conversions",
    "conversion_rate",
]
assert summary["group"].tolist() == ["control", "treatment"]
assert np.isclose(
    result.absolute_uplift,
    result.treatment_rate - result.control_rate,
)
assert np.isclose(
    result.relative_uplift,
    result.absolute_uplift / result.control_rate,
)


# Bootstrap 次数、有限性与统计范围
assert uplift_distribution.shape == (5000,)
assert np.isfinite(uplift_distribution).all()
assert np.isfinite(result.standard_error)
assert result.ci_lower <= result.ci_upper
assert 0.0 <= result.probability_positive <= 1.0
assert result.n_bootstrap == 5000
assert result.seed == 2026


# 相同随机种子必须完全复现
result_2, uplift_distribution_2 = evaluate_uplift(
    data,
    n_bootstrap=5000,
    ci_level=0.95,
    chunk_size=384,
    seed=2026,
)
assert result == result_2
assert np.array_equal(
    uplift_distribution,
    uplift_distribution_2,
)


# chunk_size 不整除 n_bootstrap 时，仍必须返回准确次数
rng = np.random.default_rng(7)
small_distribution = bootstrap_rate_differences(
    control=np.array([0, 1, 0, 1, 0], dtype=float),
    treatment=np.array([1, 1, 0, 1], dtype=float),
    n_bootstrap=23,
    chunk_size=7,
    rng=rng,
)
assert small_distribution.shape == (23,)


# 两组全部相同且为常数时，提升分布应恒为 0
rng = np.random.default_rng(9)
zero_distribution = bootstrap_rate_differences(
    control=np.zeros(8),
    treatment=np.zeros(11),
    n_bootstrap=31,
    chunk_size=6,
    rng=rng,
)
assert np.array_equal(
    zero_distribution,
    np.zeros(31),
)
```

## 3. 学习目标

完成本练习后，你应该能够：

- 区分观测提升与提升估计的不确定性；
- 对独立实验组和对照组分别进行有放回重采样；
- 用分块方式控制 Bootstrap 的峰值内存；
- 从经验分布计算标准误、百分位置信区间和正提升概率；
- 使用 `numpy.random.Generator` 保证随机实验可复现；
- 理解绝对提升与相对提升的不同解释口径。

## 4. 待完成任务

1. 实现 `validate_ab_data`：
   - 检查 `user_id、group、converted` 列存在；
   - DataFrame 非空且不含缺失值；
   - 只允许 `control` 与 `treatment` 两个组；
   - 每个组至少有 2 个用户；
   - `user_id` 全局唯一；
   - `converted` 只能取 0 或 1。

2. 实现 `summarize_conversion`：
   - 分别计算两组用户数、转化数和转化率；
   - 组别顺序固定为对照组在前；
   - 返回固定列顺序；
   - 数量列为整数，转化率为浮点数。

3. 实现 `bootstrap_rate_differences`：
   - 检查两组是一维、非空数组；
   - `n_bootstrap` 与 `chunk_size` 必须为正整数；
   - 每块使用整数索引分别对两组有放回采样；
   - 两组抽样过程相互独立；
   - 每次差值为 `treatment_mean - control_mean`；
   - 最终结果顺序与生成顺序一致，长度严格等于 `n_bootstrap`；
   - 不在函数内部重新创建或重置随机数生成器。

4. 实现 `evaluate_uplift`：
   - `ci_level` 必须位于 `(0, 1)`；
   - 先调用输入校验；
   - 计算观测绝对提升与相对提升；
   - 若对照组观测转化率为 0，明确抛出异常；
   - 用指定 `seed` 创建随机数生成器；
   - 标准误使用 `ddof=1`；
   - 根据 `ci_level` 计算两侧等尾百分位区间；
   - 所有结果字段转换为 Python 标量。

5. 运行全部检查，并思考：
   - 为什么不能把实验组和对照组混合后统一重采样；
   - 置信区间跨过 0 时，应该如何向业务解释；
   - 用户可能重复访问或属于同一门店时，“用户独立”假设会怎样失效；
   - 若需要保留城市或新老客结构，应如何改造成分层 Bootstrap。
