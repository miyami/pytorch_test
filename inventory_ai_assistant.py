"""库存及订货 AI 咨询助理（离线版）

一个可直接运行的 Python 原型：
- 输入库存、需求、供应商参数
- 输出补货点、安全库存、EOQ 以及自然语言建议
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from typing import Any


SERVICE_LEVEL_Z = {
    0.80: 0.84,
    0.85: 1.04,
    0.90: 1.28,
    0.95: 1.65,
    0.98: 2.05,
    0.99: 2.33,
}


@dataclass
class SKUProfile:
    sku: str
    unit_price: float
    annual_holding_rate: float = 0.2


@dataclass
class DemandForecast:
    avg_daily_demand: float
    daily_demand_std: float
    annual_demand: float


@dataclass
class SupplierTerms:
    lead_time_days: float
    order_cost: float
    moq: float = 0.0


@dataclass
class InventorySnapshot:
    on_hand: float
    on_order: float
    backorder: float = 0.0


class InventoryAdvisor:
    """基于运营管理经典公式 + 规则引擎的咨询助手。"""

    def __init__(self, service_level: float = 0.95):
        self.service_level = min(SERVICE_LEVEL_Z.keys(), key=lambda x: abs(x - service_level))
        self.z_score = SERVICE_LEVEL_Z[self.service_level]

    def safety_stock(self, demand: DemandForecast, supplier: SupplierTerms) -> float:
        return self.z_score * demand.daily_demand_std * math.sqrt(max(supplier.lead_time_days, 0.0))

    def reorder_point(self, demand: DemandForecast, supplier: SupplierTerms) -> float:
        expected_lead_time_demand = demand.avg_daily_demand * supplier.lead_time_days
        return expected_lead_time_demand + self.safety_stock(demand, supplier)

    def eoq(self, profile: SKUProfile, demand: DemandForecast, supplier: SupplierTerms) -> float:
        h = max(profile.unit_price * profile.annual_holding_rate, 1e-6)
        q = math.sqrt((2 * demand.annual_demand * supplier.order_cost) / h)
        return max(q, supplier.moq)

    def inventory_position(self, inv: InventorySnapshot) -> float:
        return inv.on_hand + inv.on_order - inv.backorder

    def stock_health(self, inv_position: float, reorder_point: float) -> str:
        if inv_position < reorder_point * 0.8:
            return "高风险缺货"
        if inv_position < reorder_point:
            return "接近补货点"
        if inv_position > reorder_point * 2.2:
            return "库存偏高"
        return "库存健康"

    def generate_recommendations(
        self,
        profile: SKUProfile,
        demand: DemandForecast,
        supplier: SupplierTerms,
        inv: InventorySnapshot,
    ) -> dict[str, Any]:
        ss = self.safety_stock(demand, supplier)
        rop = self.reorder_point(demand, supplier)
        eoq_qty = self.eoq(profile, demand, supplier)
        inv_pos = self.inventory_position(inv)
        health = self.stock_health(inv_pos, rop)

        suggestions: list[str] = []
        if inv_pos <= rop:
            suggestions.append(
                f"建议立即下单，建议订货量约 {math.ceil(eoq_qty)} 件（已考虑 MOQ={supplier.moq:g}）。"
            )
        if health == "高风险缺货":
            suggestions.append("缺货风险高：可启用加急采购、临时替代品或渠道调拨。")
        if health == "库存偏高":
            suggestions.append("库存偏高：可降低下一次订货量，或配合促销/捆绑销售去库存。")

        lead_time_impact = ""
        if supplier.lead_time_days > 14:
            lead_time_impact = "供应周期偏长，建议与供应商谈判缩短交期，或提升安全库存服务水平。"
        elif supplier.lead_time_days < 5:
            lead_time_impact = "供应周期较短，可适度降低安全库存以释放资金占用。"

        if lead_time_impact:
            suggestions.append(lead_time_impact)

        if not suggestions:
            suggestions.append("当前参数下运行稳定，建议每周复盘一次需求波动并动态更新参数。")

        return {
            "sku": profile.sku,
            "service_level": self.service_level,
            "inventory_position": round(inv_pos, 2),
            "safety_stock": round(ss, 2),
            "reorder_point": round(rop, 2),
            "recommended_order_qty": math.ceil(eoq_qty),
            "stock_health": health,
            "recommendations": suggestions,
        }


def build_demo_input() -> dict[str, Any]:
    return {
        "profile": {"sku": "SKU-1001", "unit_price": 85, "annual_holding_rate": 0.22},
        "demand": {"avg_daily_demand": 45, "daily_demand_std": 12, "annual_demand": 16425},
        "supplier": {"lead_time_days": 10, "order_cost": 260, "moq": 300},
        "inventory": {"on_hand": 320, "on_order": 80, "backorder": 0},
    }


def run_assistant(payload: dict[str, Any], service_level: float = 0.95) -> dict[str, Any]:
    advisor = InventoryAdvisor(service_level=service_level)
    profile = SKUProfile(**payload["profile"])
    demand = DemandForecast(**payload["demand"])
    supplier = SupplierTerms(**payload["supplier"])
    inv = InventorySnapshot(**payload["inventory"])
    return advisor.generate_recommendations(profile, demand, supplier, inv)


def main() -> None:
    parser = argparse.ArgumentParser(description="库存及订货 AI 咨询助理")
    parser.add_argument("--input", type=str, help="JSON 文件路径")
    parser.add_argument("--service-level", type=float, default=0.95, help="目标服务水平，如 0.95")
    parser.add_argument("--demo", action="store_true", help="运行内置示例")
    args = parser.parse_args()

    if args.demo:
        payload = build_demo_input()
    elif args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        raise SystemExit("请使用 --demo 或 --input <file.json>")

    result = run_assistant(payload, service_level=args.service_level)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
