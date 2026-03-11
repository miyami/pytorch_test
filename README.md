# 库存及订货 AI 咨询助理

这是一个轻量级 Python 原型，用于快速给出库存与订货建议。

## 能力
- 计算安全库存（Safety Stock）
- 计算补货点（Reorder Point）
- 计算经济订货批量（EOQ）
- 输出中文策略建议（缺货风险、库存偏高、交期优化等）

## 快速开始
```bash
python inventory_ai_assistant.py --demo
```

使用外部 JSON：
```bash
python inventory_ai_assistant.py --input sample_input.json --service-level 0.95
```

## 输入 JSON 结构
```json
{
  "profile": {"sku": "SKU-1001", "unit_price": 85, "annual_holding_rate": 0.22},
  "demand": {"avg_daily_demand": 45, "daily_demand_std": 12, "annual_demand": 16425},
  "supplier": {"lead_time_days": 10, "order_cost": 260, "moq": 300},
  "inventory": {"on_hand": 320, "on_order": 80, "backorder": 0}
}
```

## 测试
```bash
python -m unittest discover -s tests -p 'test_*.py'
```
