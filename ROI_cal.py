import pandas as pd

# Load datasets
solar_gen_df = pd.read_csv("data/time_series_60min_singleindex.csv")
seg_tariff_df = pd.read_csv("data/Monthly_SEG_FIT_Export_Tariffs__2015_2020_.csv")
cost_df = pd.read_csv("data/Monthly_Solar_Cost_per_kW__2015_2020_.csv")

# Prepare solar generation data
solar_gen_df['utc_timestamp'] = pd.to_datetime(solar_gen_df['utc_timestamp'])
solar_gen_df['date'] = solar_gen_df['utc_timestamp'].dt.date

daily_gen = solar_gen_df.groupby('date')['GB_UKM_solar_generation_actual'].sum().reset_index()
daily_cap = solar_gen_df.groupby('date')['GB_UKM_solar_capacity'].mean().reset_index()
daily_gen.rename(columns={'GB_UKM_solar_generation_actual': 'daily_generation_kWh'}, inplace=True)
daily_cap.rename(columns={'GB_UKM_solar_capacity': 'daily_capacity_kW'}, inplace=True)

normalized_df = pd.merge(daily_gen, daily_cap, on='date')
normalized_df['generation_per_kW'] = normalized_df['daily_generation_kWh'] / normalized_df['daily_capacity_kW']
normalized_df['scaled_generation_kWh'] = normalized_df['generation_per_kW'] * 1000  # For 1 MW plant

# Prepare SEG tariff data
seg_tariff_df['Month'] = pd.to_datetime(seg_tariff_df['Month'])
seg_tariff_df['Tariff_GBP'] = seg_tariff_df['Export_Tariff_p_kWh'] / 100  # Convert to GBP
seg_tariff_daily = seg_tariff_df[['Month', 'Tariff_GBP']].rename(columns={'Month': 'date'})
seg_tariff_daily = seg_tariff_daily.set_index('date').resample('D').ffill().reset_index()

# Prepare cost data
cost_df['date'] = pd.to_datetime(cost_df[['Year', 'Month']].assign(DAY=1))
cost_df_daily = cost_df[['date', 'Cost_per_kW_GBP']]
cost_df_daily = cost_df_daily.set_index('date').resample('D').ffill().reset_index()

# Merge all data
normalized_df['date'] = pd.to_datetime(normalized_df['date'])
normalized_df = normalized_df.merge(seg_tariff_daily, on='date', how='left')
normalized_df = normalized_df.merge(cost_df_daily, on='date', how='left')

# Drop missing values
normalized_df = normalized_df.dropna(subset=['generation_per_kW', 'Tariff_GBP', 'Cost_per_kW_GBP'])

# Calculate revenue and ROI
investment = 1800 * 1000  # £1800 per kW * 1000 kW = £1.8M
normalized_df['revenue_GBP'] = normalized_df['scaled_generation_kWh'] * normalized_df['Tariff_GBP']
normalized_df['cumulative_revenue_GBP'] = normalized_df['revenue_GBP'].cumsum()
normalized_df['ROI_percent'] = (normalized_df['cumulative_revenue_GBP'] / investment) * 100

# Save full data to CSV
normalized_df.to_csv("solar_plant_normalized_ROI.csv", index=False)

# ROI after 5 years
five_years = normalized_df[normalized_df['date'] <= pd.to_datetime("2019-12-31")]
final_cum_revenue = five_years['cumulative_revenue_GBP'].iloc[-1]
final_roi = five_years['ROI_percent'].iloc[-1]

print(f"Cumulative Revenue after 5 years: £{final_cum_revenue:,.2f}")
print(f"ROI after 5 years: {final_roi:.2f}%")
print("Full ROI data saved to: solar_plant_normalized_ROI.csv")
