# -*- coding: utf-8 -*-
"""
Inventory Policy Module
------------------------------------------------------------
Recommends safety stock and reorder points using target service
levels, demand variability, and lead time uncertainty.
Outputs:
  - recommended_safety_stock
  - recommended_reorder_point
  - implied_fill_rate
  - sensitivity table
------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import os

# ======================================================================
# --- USER CONFIGURATION ---
# ======================================================================
PRODUCT_TO_ANALYZE = 'Pharmaceuticals'       # product_category from your data
SUPPLIER_TO_ANALYZE = 'SUP_00001'           # supplier_id from your data
TARGET_SERVICE_LEVEL = 0.995                # e.g., 99.5% service level
# ======================================================================


def analyze_inventory_drivers(shipments_df, suppliers_df):
    """
    Analyzes demand and lead-time variability to derive key inventory drivers.
    """
    print("Step 1: Analyzing historical data to determine inventory drivers...")

    # --- Lead Time Calculation (from Supplier data) ---
    print("-> Using 'lead_time_days' and 'delivery_performance' from supplier file.")
    lead_time_stats = suppliers_df[['supplier_id', 'lead_time_days', 'delivery_performance']].copy()
    lead_time_stats.rename(columns={'lead_time_days': 'avg_lead_time'}, inplace=True)

    # Estimate std deviation of lead time based on delivery performance
    lead_time_stats['std_dev_lead_time'] = (100 - lead_time_stats['delivery_performance']) / 10

    # --- Demand Calculation (from Shipments data) ---
    print("-> Shipments file has no explicit date. Simulating shipment days for variability analysis.")
    shipments_df = shipments_df.copy()
    shipments_df['simulated_date'] = pd.date_range(start='2024-01-01', periods=len(shipments_df), freq='D')

    # Group by simulated date and product_category to get daily demand
    daily_demand = shipments_df.groupby(
        [shipments_df['simulated_date'].dt.date, 'product_category']
    )['weight_kg'].sum().reset_index()

    demand_stats = daily_demand.groupby('product_category')['weight_kg'].agg(['mean', 'std']).reset_index()
    demand_stats.rename(columns={'mean': 'avg_daily_demand', 'std': 'std_dev_daily_demand'}, inplace=True)
    demand_stats['std_dev_daily_demand'] = demand_stats['std_dev_daily_demand'].fillna(0)

    print("-> Demand variability calculated per product category.")
    return demand_stats, lead_time_stats


def calculate_inventory_policy(avg_demand, std_dev_demand, avg_lead_time, std_dev_lead_time, service_level):
    """
    Calculates the recommended Safety Stock (SS) and Reorder Point (ROP).
    """
    z_score = norm.ppf(service_level)

    demand_variance = std_dev_demand ** 2
    lead_time_variance = std_dev_lead_time ** 2

    std_dev_during_lead_time = np.sqrt(
        (avg_lead_time * demand_variance) + ((avg_demand ** 2) * lead_time_variance)
    )

    safety_stock = z_score * std_dev_during_lead_time
    demand_during_lead_time = avg_demand * avg_lead_time
    reorder_point = demand_during_lead_time + safety_stock

    return np.ceil(safety_stock), np.ceil(reorder_point)


# ======================================================================
# --- MAIN SCRIPT EXECUTION ---
# ======================================================================
if __name__ == "__main__":
    print("\n--- Data-Driven Inventory Policy Recommendation ---")

    try:
        # CORRECTED PATHS: Point to the 'dataset' folder
        shipments_path = "dataset/project1_shipments.csv"
        suppliers_path = "dataset/project1_suppliers.csv"
        shipments = pd.read_csv(shipments_path)
        suppliers = pd.read_csv(suppliers_path)
        print(f"Data loaded successfully from '{shipments_path}' and '{suppliers_path}'.\n")
    except FileNotFoundError:
        print("ERROR: CSV files not found. Ensure they exist in the 'dataset/' folder.")
        exit()

    try:
        demand_stats, lead_time_stats = analyze_inventory_drivers(shipments, suppliers)

        product_demand = demand_stats[demand_stats['product_category'] == PRODUCT_TO_ANALYZE].iloc[0]
        supplier_lead_time = lead_time_stats[lead_time_stats['supplier_id'] == SUPPLIER_TO_ANALYZE].iloc[0]

    except (KeyError, IndexError) as e:
        print(f"\nERROR: Could not find '{PRODUCT_TO_ANALYZE}' or '{SUPPLIER_TO_ANALYZE}' in the data.")
        print("Please check the configuration variables at the top of the script.")
        exit()

    # Extract key parameters
    avg_d = product_demand['avg_daily_demand']
    std_d = product_demand['std_dev_daily_demand']
    avg_l = supplier_lead_time['avg_lead_time']
    std_l = supplier_lead_time['std_dev_lead_time']

    print("----------------------------------------------------")
    print(f"Generating Policy for '{PRODUCT_TO_ANALYZE}' from '{SUPPLIER_TO_ANALYZE}'...")
    print("----------------------------------------------------")

    ss, rop = calculate_inventory_policy(avg_d, std_d, avg_l, std_l, TARGET_SERVICE_LEVEL)

    print(f"\n Inventory Policy Summary ({TARGET_SERVICE_LEVEL:.1%} Service Level):")
    print(f"  • Average Daily Demand: {avg_d:.2f} kg")
    print(f"  • Average Lead Time: {avg_l:.2f} days")
    print(f"  • Demand Std Dev: {std_d:.2f} kg/day")
    print(f"  • Lead Time Std Dev: {std_l:.2f} days")
    print(f"\n  → Recommended Safety Stock: {ss:.0f} kg")
    print(f"  → Recommended Reorder Point: {rop:.0f} kg")
    print(f"  → Implied Fill Rate: {TARGET_SERVICE_LEVEL:.1%}")

    # --- Sensitivity Analysis ---
    print("\n--- Sensitivity Analysis Table ---")
    sensitivity_levels = [0.90, 0.95, 0.975, 0.99, 0.995]
    table_data = []

    for level in sensitivity_levels:
        ss_level, rop_level = calculate_inventory_policy(avg_d, std_d, avg_l, std_l, level)
        table_data.append({
            "Target Service Level": f"{level:.1%}",
            "Safety Stock (kg)": f"{ss_level:.0f}",
            "Reorder Point (kg)": f"{rop_level:.0f}"
        })

    sensitivity_df = pd.DataFrame(table_data)
    print(sensitivity_df.to_string(index=False))

    # Save the output to a file
    output_dir = "inventory_policy"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "inventory_policy_sensitivity.csv")
    sensitivity_df.to_csv(output_path, index=False)
    print(f"\n-> Sensitivity analysis saved to '{output_path}'")

    print("\n Analysis complete. Results ready for reporting.")