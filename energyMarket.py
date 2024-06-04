# Author: Shaun Sweeney 
# Contact: shaunsweeney12@gmail.com
# Initial creation: December 2022

# Purpose: 

# This script uses config settings set in configInputs and begins the market instance. Most of the heavy lifting is done in the marketFunctions file. 


import time
import pandas as pd
import numpy as np
import marketFunctions as mf
import marketDesignFunctions as mdf
import datetime as dt
import pickle
pd.options.mode.chained_assignment = None  # default='warn'



def allocateEnergy(execution_type_parameters, execution_times, neighbours_list, requests, neighbours_consumption_ts_list, consumption_prediction, supply_input, pricing_function_parameters, scaling_methodology, flex_level_key):

    market_parameters = mdf.defaultMarketDesign(execution_times)
    execution_days = (execution_times.get("end_time") - execution_times.get("start_time")).days 
    execution_parameters = mf.getExecutionParametersNew(execution_times.get("start_time"), execution_times.get("end_time"), market_parameters)
    time_series = np.arange(0, execution_parameters.get("num_execution_samples")*market_parameters.get("market_resolution"), 
                            market_parameters.get("market_resolution")); 
    ts_time_series = time_series[0:execution_parameters.get("num_execution_samples")]
    dt_time_series = pd.date_range(start=execution_times.get("start_time"), end=execution_times.get("end_time")-dt.timedelta(seconds=1), freq=market_parameters.get("market_resolution"))
    initial_power_and_cost_values = mf.getInitialPowerAndCostValues(execution_type_parameters.get("essential_consumption_type"), execution_parameters, dt_time_series, neighbours_list, neighbours_consumption_ts_list, supply_input, consumption_prediction, time_series, ts_time_series, pricing_function_parameters)
   
    booked_flexible_consumption = pd.Series(np.zeros(execution_parameters.get("num_execution_samples")), index = dt_time_series)
    ts_initial_available_flexible_supply = initial_power_and_cost_values['flexible supply']
    ts_initial_available_flexible_supply.index = dt_time_series
    ts_available_flexible_supply = ts_initial_available_flexible_supply
    ts_grid_health = initial_power_and_cost_values['grid_health']
    ts_grid_health.index = dt_time_series
    ts_energy_buy_price = initial_power_and_cost_values['buy_price']
    ts_energy_buy_price.index = dt_time_series
    ts_energy_sell_price = initial_power_and_cost_values['sell_price']
 
    market_iter=0

    market_timings = mf.setMarketTimings(market_parameters) 
    requests.sort(key=lambda request: request.earliest_start_time, reverse=False) 


    if(execution_type_parameters.get("resource_allocation_method") == "queue"):
        mf.allocateEnergyAccordingToQueues(execution_type_parameters, execution_times, execution_parameters, market_parameters, pricing_function_parameters, market_timings, market_iter, dt_time_series, initial_power_and_cost_values, consumption_prediction, requests, neighbours_list, ts_available_flexible_supply, ts_grid_health, ts_energy_buy_price, ts_energy_sell_price, booked_flexible_consumption, scaling_methodology, flex_level_key)
    elif((execution_type_parameters.get("resource_allocation_method") == "volumeoptimisation") or (execution_type_parameters.get("resource_allocation_method") == "priceoptimisation")):
        mf.generalOptimisation(execution_type_parameters, execution_times, market_timings, market_parameters, ts_available_flexible_supply, neighbours_list, requests, scaling_methodology, flex_level_key)



