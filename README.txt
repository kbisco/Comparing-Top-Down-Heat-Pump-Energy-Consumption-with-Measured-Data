# Comparing-Top-Down-Heat-Pump-Energy-Consumption-with-Measured-Data
Data:
Indiana hourly, census tract temperature data (Source: temperature from MERRA-2, census tract boundaries from US Census shapefiles) 
  ct_temps_IN_2010.gz, ct_temps_IN_2020.gz, ct_temps_IN_2021.gz, ct_temps_IN_2022.gz
Indiana cooling degree days by census tract (Source: NOAA)
  CDD_ct_IN.csv
Indiana total housing area in square feet (Source: FEMA Hazus)
  IN_sqft_HAZUS.xls
Indiana retail sales of electricity in million kWh (Source: EIA)
  Retail_sales_of_electricity_IN.csv
  (Note: other consumption data is hard-coded into respective scripts)
Model weight for heating-related fossil fuel consumption (obtained from ff_demand_IN.py)
  heat_weight_ff_INct.csv
Model weight for heating-related electricity consumption (obtained from elec_demand_IN.py)
  heat_weight_elec_INct.csv
Model weight for cooling-related electricity consumption (obtained from elec_demand_IN.py)
  cool_weight_elec_INct.csv

Scripts:
