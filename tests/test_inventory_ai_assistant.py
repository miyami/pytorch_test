import unittest

from inventory_ai_assistant import (
    InventoryAdvisor,
    SKUProfile,
    DemandForecast,
    SupplierTerms,
    InventorySnapshot,
    run_assistant,
    build_demo_input,
)


class TestInventoryAssistant(unittest.TestCase):
    def setUp(self):
        self.profile = SKUProfile(sku="A", unit_price=100, annual_holding_rate=0.2)
        self.demand = DemandForecast(avg_daily_demand=20, daily_demand_std=5, annual_demand=7300)
        self.supplier = SupplierTerms(lead_time_days=7, order_cost=200, moq=100)
        self.inv = InventorySnapshot(on_hand=60, on_order=20, backorder=0)
        self.advisor = InventoryAdvisor(service_level=0.95)

    def test_core_metrics_positive(self):
        ss = self.advisor.safety_stock(self.demand, self.supplier)
        rop = self.advisor.reorder_point(self.demand, self.supplier)
        eoq = self.advisor.eoq(self.profile, self.demand, self.supplier)
        self.assertGreater(ss, 0)
        self.assertGreater(rop, ss)
        self.assertGreaterEqual(eoq, self.supplier.moq)

    def test_low_inventory_triggers_order(self):
        result = self.advisor.generate_recommendations(
            self.profile, self.demand, self.supplier, self.inv
        )
        self.assertIn(result["stock_health"], {"高风险缺货", "接近补货点"})
        self.assertTrue(any("建议立即下单" in s for s in result["recommendations"]))

    def test_demo_payload_runs(self):
        result = run_assistant(build_demo_input(), service_level=0.95)
        self.assertIn("sku", result)
        self.assertIn("recommended_order_qty", result)


if __name__ == "__main__":
    unittest.main()
