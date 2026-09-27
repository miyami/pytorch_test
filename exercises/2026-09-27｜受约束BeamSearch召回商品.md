# 2026-09-27｜受约束BeamSearch召回商品

## 1. 任务背景

昨天你已经用前缀树约束商品语义 ID（Semantic ID，简称 SID）的逐步解码，但贪心解码每一步只保留当前概率最高的 token，可能因为早期的局部最优选择而错过整体概率更高的合法商品。

今天把解码器升级为**受约束 Beam Search**：在每一步同时保留多个合法候选前缀，用累计对数概率比较完整路径。前缀树负责保证候选始终对应某个商品 SID 的合法前缀，Beam Search 负责在有限计算量下改善搜索质量。

预计完成时间：20–30 分钟。

## 2. 起始代码

```python
from typing import Dict, Iterable, List, Tuple

import torch


EOS_TOKEN = 8
VOCAB_SIZE = 9
Beam = Tuple[Tuple[int, ...], float, bool]
# Beam = (已生成 token, 累计对数概率, 是否已生成 EOS)

catalog = {
    (0, 4, EOS_TOKEN): "牛奶",
    (0, 5, EOS_TOKEN): "酸奶",
    (1, 5, EOS_TOKEN): "苹果",
    (1, 6, EOS_TOKEN): "香蕉",
}


def build_prefix_trie(
    sequences: Iterable[Tuple[int, ...]],
) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    """返回“前缀 -> 下一步合法 token”的只读式映射。"""
    next_tokens: Dict[Tuple[int, ...], set[int]] = {}

    for sequence in sequences:
        for length in range(len(sequence)):
            prefix = sequence[:length]
            next_tokens.setdefault(prefix, set()).add(sequence[length])

    return {
        prefix: tuple(sorted(tokens))
        for prefix, tokens in next_tokens.items()
    }


trie = build_prefix_trie(catalog.keys())


def make_logits(token_scores: Dict[int, float]) -> torch.Tensor:
    """构造一维 logits；未列出的 token 故意给出较大的干扰值。"""
    logits = torch.full((VOCAB_SIZE,), 7.0)
    for token, score in token_scores.items():
        logits[token] = score
    return logits


# 只有前缀树允许的 token 才能参与归一化。
# 根节点上 token 0 略占优势，但它后续的最佳分支较弱；
# token 1 的第一步略弱，却能接上一条整体概率更高的路径。
logits_by_prefix = {
    (): make_logits({0: 3.0, 1: 2.8}),
    (0,): make_logits({4: 0.0, 5: 0.0}),
    (1,): make_logits({5: 5.0, 6: 0.0}),
    (0, 4): make_logits({EOS_TOKEN: 2.0}),
    (0, 5): make_logits({EOS_TOKEN: 2.0}),
    (1, 5): make_logits({EOS_TOKEN: 2.0}),
    (1, 6): make_logits({EOS_TOKEN: 2.0}),
}


def allowed_next_tokens(
    trie: Dict[Tuple[int, ...], Tuple[int, ...]],
    prefix: Tuple[int, ...],
) -> Tuple[int, ...]:
    """读取某个前缀下一步允许生成的 token。"""
    if prefix not in trie:
        raise ValueError(f"非法或已结束的前缀: {prefix}")
    return trie[prefix]


def constrained_log_probs(
    logits: torch.Tensor,
    allowed_tokens: Tuple[int, ...],
) -> torch.Tensor:
    """
    返回长度与 logits 相同的对数概率向量。

    非法 token 的结果必须是 -inf；合法 token 之间重新做 log_softmax。
    """
    # TODO: 检查 logits 形状以及 allowed_tokens 是否为空、越界
    # TODO: 构造掩码，将非法位置设为 -inf
    # TODO: 只让合法 token 参与归一化，并返回 log_softmax 结果
    raise NotImplementedError


def beam_search(
    logits_by_prefix: Dict[Tuple[int, ...], torch.Tensor],
    trie: Dict[Tuple[int, ...], Tuple[int, ...]],
    eos_token: int,
    beam_width: int,
    max_steps: int,
) -> List[Tuple[List[int], float]]:
    """
    返回得分从高到低排列的已完成路径：
    [(token 列表, 累计对数概率), ...]
    """
    # TODO: 校验 beam_width 和 max_steps 均为正整数
    beams: List[Beam] = [((), 0.0, False)]

    # TODO: 最多执行 max_steps 轮
    # TODO: 对每个 beam：
    #       1) 已结束的 beam 原样加入候选，不能再次扩展
    #       2) 未结束的 beam 读取当前前缀的 logits 与合法 token
    #       3) 计算受约束对数概率，为每个合法 token 创建新候选
    #       4) 新得分 = 旧得分 + 当前 token 的对数概率
    # TODO: 按“得分降序、token 元组升序”进行确定性排序，只保留 beam_width 个
    # TODO: 如果保留的 beam 全部结束，则提前停止

    # TODO: 丢弃仍未生成 EOS 的路径；若没有完成路径则抛出 RuntimeError
    # TODO: 将 token 元组转为列表，按得分降序返回
    raise NotImplementedError


def decode_beams(
    beams: List[Tuple[List[int], float]],
    catalog: Dict[Tuple[int, ...], str],
) -> List[Tuple[str, float]]:
    """把合法的完整 SID 转成商品名，同时保留得分。"""
    # TODO: 校验每条路径都能在 catalog 中找到，再完成映射
    raise NotImplementedError


greedy_beams = beam_search(
    logits_by_prefix,
    trie,
    eos_token=EOS_TOKEN,
    beam_width=1,
    max_steps=3,
)
beam2 = beam_search(
    logits_by_prefix,
    trie,
    eos_token=EOS_TOKEN,
    beam_width=2,
    max_steps=3,
)

print("beam_width=1:", decode_beams(greedy_beams, catalog))
print("beam_width=2:", decode_beams(beam2, catalog))
```

## 3. 学习目标

完成练习后，你应该能够：

1. 用掩码让 `log_softmax` 只在前缀树允许的 token 上归一化。
2. 维护 Beam Search 的路径、累计对数概率和结束状态。
3. 实现候选扩展、全局排序、宽度裁剪与提前终止。
4. 处理 EOS、非法前缀、未完成路径和确定性并列排序。
5. 解释为什么较宽的 beam 可能战胜逐步贪心选择。

## 4. 待完成任务

1. 完成 `constrained_log_probs`：
   - 输入必须是一维 logits；
   - `allowed_tokens` 不能为空且不能越界；
   - 非法 token 的结果必须为 `-inf`；
   - 合法 token 的概率之和应接近 1。

2. 完成 `beam_search` 的核心循环：
   - 每轮扩展所有未结束 beam；
   - 使用累计**对数概率相加**为完整路径评分；
   - 已生成 EOS 的 beam 只能保留，不能继续扩展；
   - 每轮只保留 `beam_width` 个最佳候选；
   - 分数相同时按 token 元组升序，保证结果可复现；
   - 所有保留路径结束后提前停止。

3. 完成 `decode_beams`，把完整 SID 映射为商品名；遇到目录外路径时应明确报错。

4. 为你的实现补充并通过以下断言：

```python
log_probs = constrained_log_probs(
    logits_by_prefix[()],
    allowed_next_tokens(trie, ()),
)
assert torch.isneginf(log_probs[2])
assert torch.allclose(log_probs[[0, 1]].exp().sum(), torch.tensor(1.0))

assert greedy_beams[0][0] == [0, 4, EOS_TOKEN]
assert beam2[0][0] == [1, 5, EOS_TOKEN]
assert beam2[0][1] > greedy_beams[0][1]
assert all(tuple(tokens) in catalog for tokens, _ in beam2)
assert all(
    beam2[i][1] >= beam2[i + 1][1]
    for i in range(len(beam2) - 1)
)
```

5. 用 2–3 句话解释：为什么根节点上概率较低的 token 1，最终能在 `beam_width=2` 时得到比贪心路径更高的累计得分？
