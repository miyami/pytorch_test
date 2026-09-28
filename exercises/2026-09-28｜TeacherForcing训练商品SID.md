# 2026-09-28｜TeacherForcing训练商品SID

## 1. 任务背景

过去三天你已经完成了商品 Semantic ID（SID）的残差量化、前缀树约束和 Beam Search 解码。今天转到**模型训练侧**：给定用户或查询特征，让一个小型自回归模型学习逐 token 生成目标商品 SID。

训练时使用 Teacher Forcing：第 (t) 个位置输入真实 token，预测第 (t+1) 个 token。由于不同 SID 长度不同，批处理后会出现 `PAD`；这些补齐位置既不能参与损失，也不能计入准确率。你需要自己完成输入/标签错位、GRU 前向传播、掩码交叉熵和一次参数更新。

预计完成时间：20–30 分钟。

## 2. 起始代码

```python
from typing import Tuple

import torch
from torch import nn


torch.manual_seed(42)

# SID code token: 0～7；特殊 token: EOS=8, BOS=9, PAD=10
EOS_TOKEN = 8
BOS_TOKEN = 9
PAD_TOKEN = 10
VOCAB_SIZE = 11

# 每行可理解为一个用户/查询的稠密特征
query_features = torch.tensor(
    [
        [1.0, 0.2, -0.1, 0.4],
        [0.1, 1.1, 0.3, -0.2],
        [0.8, -0.4, 0.9, 0.1],
        [-0.2, 0.5, 1.0, 0.7],
    ],
    dtype=torch.float32,
)

# 序列已经包含 BOS 和 EOS；较短序列在 EOS 后用 PAD 补齐
target_sequences = torch.tensor(
    [
        [BOS_TOKEN, 0, 4, EOS_TOKEN, PAD_TOKEN],
        [BOS_TOKEN, 1, 5, 2, EOS_TOKEN],
        [BOS_TOKEN, 2, 3, EOS_TOKEN, PAD_TOKEN],
        [BOS_TOKEN, 1, 6, 7, EOS_TOKEN],
    ],
    dtype=torch.long,
)


def make_teacher_forcing_batch(
    sequences: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 [BOS, s1, s2, EOS, PAD] 拆成：
    decoder_input = [BOS, s1, s2, EOS]
    labels        = [s1,  s2, EOS, PAD]
    """
    # TODO: 检查 sequences 必须是二维 LongTensor，且至少有 2 列
    # TODO: 使用切片构造错开一位的 decoder_input 和 labels
    raise NotImplementedError


class SIDGenerator(nn.Module):
    def __init__(
        self,
        query_dim: int,
        vocab_size: int,
        embedding_dim: int = 8,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_TOKEN,
        )
        self.query_to_hidden = nn.Linear(query_dim, hidden_dim)
        self.decoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        query: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数：
            query: [batch_size, query_dim]
            decoder_input: [batch_size, seq_len]
        返回：
            logits: [batch_size, seq_len, vocab_size]
        """
        # TODO: 检查 batch_size 是否一致
        # TODO: token embedding，得到 [B, T, embedding_dim]
        # TODO: 将 query 投影为 GRU 初始隐状态 [1, B, hidden_dim]
        # TODO: 执行 GRU，并将每个时间步的输出映射到词表 logits
        raise NotImplementedError


def masked_sequence_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_token: int,
) -> torch.Tensor:
    """
    计算非 PAD token 的平均交叉熵。

    不要直接使用默认 reduction="mean"；请先求有效 token 的损失总和，
    再除以有效 token 数量，使分母逻辑清晰可检查。
    """
    # TODO: 检查 logits 与 labels 的前三个相关维度是否匹配
    # TODO: 将 [B, T, V] 和 [B, T] 展平
    # TODO: 使用 ignore_index=pad_token、reduction="sum" 计算损失总和
    # TODO: 统计 labels 中非 PAD token 数；若为 0 应报错
    # TODO: 返回 loss_sum / valid_token_count
    raise NotImplementedError


def masked_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_token: int,
) -> torch.Tensor:
    """返回非 PAD 位置上的 token 准确率。"""
    # TODO: 对词表维度取 argmax
    # TODO: 构造有效位置 mask，只在 mask 内统计正确数与总数
    raise NotImplementedError


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    query: torch.Tensor,
    sequences: torch.Tensor,
) -> Tuple[float, float]:
    """完成一次 Teacher Forcing 参数更新，返回 loss 和 token accuracy。"""
    # TODO: 构造 decoder_input 与 labels
    # TODO: 切换训练模式并清空梯度
    # TODO: 前向传播、计算掩码损失、反向传播、更新参数
    # TODO: 在不记录梯度的情况下计算准确率
    # TODO: 返回 Python float
    raise NotImplementedError


decoder_input, labels = make_teacher_forcing_batch(target_sequences)

model = SIDGenerator(
    query_dim=query_features.shape[1],
    vocab_size=VOCAB_SIZE,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

loss, accuracy = train_one_step(
    model,
    optimizer,
    query_features,
    target_sequences,
)

print("decoder_input shape:", decoder_input.shape)
print("labels shape:", labels.shape)
print(f"loss={loss:.4f}, token_accuracy={accuracy:.4f}")
```

## 3. 学习目标

完成练习后，你应该能够：

1. 用序列切片构造 Teacher Forcing 的解码输入与下一 token 标签。
2. 把查询特征转换为 GRU 的初始隐状态，并理解 `[B, T, V]` 的 logits 形状。
3. 使用 `ignore_index` 排除 PAD，并以有效 token 数作为损失分母。
4. 用同一有效位置掩码计算 token accuracy，避免补齐位置虚增指标。
5. 正确组织 PyTorch 的一次训练步骤：清梯度、前向、反向和参数更新。

## 4. 待完成任务

1. 完成 `make_teacher_forcing_batch`，确认输入与标签严格错开一位，并且不修改原始张量。

2. 完成 `SIDGenerator.forward`：
   - token embedding 的形状为 `[B, T, embedding_dim]`；
   - 查询特征投影后增加 GRU 层数维，形成 `[1, B, hidden_dim]`；
   - 最终 logits 形状必须为 `[B, T, VOCAB_SIZE]`。

3. 完成 `masked_sequence_loss` 和 `masked_token_accuracy`：
   - PAD 位置完全不参与损失和准确率；
   - 如果所有标签都是 PAD，应明确抛出异常；
   - 不要在计算损失前手动删除 batch 或时间维。

4. 完成 `train_one_step`，确保参数确实得到更新，并且返回普通 Python `float`。

5. 补充并通过以下检查：

```python
assert decoder_input.shape == (4, 4)
assert labels.shape == (4, 4)
assert torch.equal(decoder_input[:, 1:], target_sequences[:, 1:-1])
assert torch.equal(labels, target_sequences[:, 1:])

with torch.no_grad():
    logits = model(query_features, decoder_input)

assert logits.shape == (4, 4, VOCAB_SIZE)

loss_before = masked_sequence_loss(logits, labels, PAD_TOKEN)
changed_logits = logits.clone()
changed_logits[labels == PAD_TOKEN] = 999.0
loss_after = masked_sequence_loss(changed_logits, labels, PAD_TOKEN)
assert torch.allclose(loss_before, loss_after)

accuracy = masked_token_accuracy(logits, labels, PAD_TOKEN)
assert 0.0 <= accuracy.item() <= 1.0
```

6. 用 2～3 句话回答：如果把 PAD 位置也计入损失和准确率，模型训练与指标分别可能出现什么误导？
