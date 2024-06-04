# Author: Shaun Sweeney 
# Contact: shaunsweeney12@gmail.com
# Initial creation: June 2023

# Purpose: 

# This script is used to relate the supply dataset to the consumption dataset under a number of different methodologies. 


import pandas as pd
import datetime as dt


def standardiseSupplyInputs(start_time, end_time, number_execution_households, scaling_methodology, neighbours_list, neighbours_consumption_ts_list, requests):

    convert_mw_to_kw = 1e3 #convert mw to kw
    convert_watts_to_kw=1e3
    num_seconds_hour=60*60
    
    #read historic GB supply data - this is in MW - consumption data is assumed to be in kW
    df = pd.read_csv('files/df_fuel_ckan.csv', parse_dates=['DATETIME']) #source of data - #https://data.nationalgrideso.com/carbon-intensity1/historic-generation-mix/r/historic_gb_generation_mix
    df.set_index('DATETIME', inplace=True)

    #get subset of renewables data from overall dataset
    df_renewables = df['WIND'] + df['HYDRO'] + df['SOLAR']

    #remove the dates as we do not have corresponding dates for consumption
    df_renewables = df_renewables.tz_convert(None)

    total_supply_ts = df_renewables.loc[start_time:end_time-dt.timedelta(seconds=1)]*convert_mw_to_kw
    # POSSIBLE TO DO: it may be desired to derate the supply dataset by creating a timeseries of the isntalled capacity, generation capacity register https://connecteddata.nationalgrid.co.uk/dataset/generation-capacity-register


    #### 3 scaling methodologies below #####

    # 1. Scale the supply dataset so that it can meet an (approximate) percentage of the consumption
    if(scaling_methodology.get("method")=='scale_to_match_consumption'):
        total_essential_consumption_required = 0
        for neighbour_id, neighbour in enumerate(neighbours_consumption_ts_list):
            start_time = neighbour[1]['essential'].index[0]
            end_time = neighbour[1]['essential'].index[-1]
            duration = end_time-start_time
            total_essential_consumption_required += duration.total_seconds()/num_seconds_hour*sum(neighbour[1]['essential']/convert_watts_to_kw)


        total_flexible_consumption_required=0
        for request in requests:
            total_flexible_consumption_required += request.duration.total_seconds()/num_seconds_hour*request.power/convert_watts_to_kw

        total_consumption_required = total_essential_consumption_required + total_flexible_consumption_required

        print('Total essential consumption required: '+str(total_essential_consumption_required))
        print('Total flexible consumption required: '+str(total_flexible_consumption_required))
        print('Total consumption required: '+str(total_consumption_required))
        print('Total proportion of consumption that is essential: '+str(total_essential_consumption_required/total_consumption_required*100)+'%')

        total_supply_sum = sum(total_supply_ts)
        supply_scaling_factor = total_consumption_required/total_supply_sum


    # 2. Scale from an assumed renewable penetration level and assumed number of households served to a new penetration level and the number of households represented in the dataset
    elif(scaling_methodology.get("method")=='scale_to_installed_capacity'):
        renewables_penetration_2022 = 0.386 #https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1107456/Energy_Trends_September_2022.pdf
        num_uk_households = 28.0e6
        supply_scaling_factor = (convert_mw_to_kw*number_execution_households/num_uk_households)/renewables_penetration_2022
  

    # 3. Scale all consumption by a fixed value 
    elif(scaling_methodology.get("method")=='scale_to_fixed_value'):
        supply_scaling_factor = 1/scaling_methodology.get("value") #units were already changed to kw in coming up with this value so no need to multiply by units change
        

    scaled_supply_ts = round(total_supply_ts*supply_scaling_factor, 2)
    scaled_supply_ts.name = 'supply'

    return scaled_supply_ts
