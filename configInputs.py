
# Author: Shaun Sweeney 
# Contact: shaunsweeney12@gmail.com
# Initial creation: December 2022

# Purpose: 

# This script sets configuration options in relation to how the market should work.


import datetime as dt
import pandas as pd
import pricingFunctions
import energyMarket
import generateSimulatedConsumption as simCon
import standardiseInputs as sI
import cProfile
import pstats
import io



############################### Define inputs ####################################


start_time = dt.datetime(2021, 12, 17)  #these dates are the dates we have moixa data available for
end_time = dt.datetime(2021, 12, 18)

execution_times = {"start_time": start_time,
                    "end_time": end_time}



execute_multiple_flex_levels=False
repeat_same_day_multiple_times=False
number_of_daily_repeats=1000

if(execute_multiple_flex_levels == True):

        #Update folder and file name
        rootfolder = 'SaveFeb2024/varyingInitFairness/'
        request_filename_0_flex=rootfolder+'moixarequests0flexlevel1713639126.pickle'
        requests_0_flex =  pd.read_pickle(r'files/'+request_filename_0_flex)
        
        #change this dict based on the number of flex levels you are using
        requests_flex_levels_dict = dict({"flex_0": requests_0_flex})

        # requests_flex_levels_dict = dict({"flex_0": requests_0_flex, 
        #         "flex_3": requests_3_flex,
        #         "flex_6": requests_6_flex,
        #         "flex_12": requests_12_flex})
        
else:
        rootfolder = 'SaveFeb2024/biasedBuyAndFairness/1initWillingessToPay/'
        request_filename = rootfolder+'moixarequests0flexlevel1717365775.pickle'
        requests_0_flex = pd.read_pickle(r'files/'+request_filename)
        requests_flex_levels_dict = dict({"flex_0": requests_0_flex})


neighbour_filename=rootfolder+'moixaneighbours1717365775.pickle'

neighbours_list = pd.read_pickle(r'files/'+neighbour_filename)
number_execution_households=len(neighbours_list)


execution_type_parameters={
"appliances_labelled":False, #change this to true if appliances are labelled - this affects how fairness scores are updated for app;liacnes in the updateFairnessMetrics function in the marketFunctions file
"essential_consumption_type":"none", #change this to "timeseries"
"use_products":"no", # change this to yes if offering different reliability levels (premium, basic) for flexible consumption
# "resource_allocation_method": "queue", #queuing with fairness
"resource_allocation_method": "volumeoptimisation", #volume maximisation
# "resource_allocation_method": "priceoptimisation", #revenue maximisation
"repeat_same_day_multiple_times": repeat_same_day_multiple_times, #this is useful for experimentation with the queuing approach which is stochastic and to see how outcomes evolve over time while removing supply variability over time
"number_of_daily_repeats": number_of_daily_repeats #for use in combination with
}



#update the filepath to point to the consumption prediction file
neighbours_consumption_ts_list=[] ####this is for storing existing categorised time series data
consumption_ts_folder='moixa/characterised/csvs/Save/'
consumption_prediction_file='files/Save/totalconsumption.csv'
consumption_prediction = pd.read_csv(consumption_prediction_file, index_col=0, parse_dates=True)



if(execution_type_parameters.get("essential_consumption_type")=="timeseries"):
        for current_neighbour_id, current_neighbour in enumerate(neighbours_list):
                print('reading consumption ts for neighbour '+str(current_neighbour_id))
                current_neighbour_consumption_ts_list = []
                current_neighbour_consumption_ts_list.append(current_neighbour_id)

                path=consumption_ts_folder+'consumptioncharacterisation_n{}'.format(current_neighbour_id)+'.csv'
                current_neighbour_consumption_ts = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
                current_neighbour_consumption_ts_list.append(current_neighbour_consumption_ts)

                neighbours_consumption_ts_list.append(current_neighbour_consumption_ts_list)



### Config for supply input
supply_input_filename = '' #TO DO: enter filename for supply data
scaling_methodology = {"method": 'scale_to_fixed_value',
"value": 250000} #found this value in supplyScenarioGraphs.ipynb



if(execute_multiple_flex_levels == True):
        supply_input = sI.standardiseSupplyInputs(start_time, end_time, number_execution_households, scaling_methodology, neighbours_list, neighbours_consumption_ts_list, requests_flex_levels_dict.get("flex_0"))
else:
        supply_input = sI.standardiseSupplyInputs(start_time, end_time, number_execution_households, scaling_methodology, neighbours_list, neighbours_consumption_ts_list, requests_flex_levels_dict.get("flex_0"))


# print('Do you wish to see a graph showing the market evolution real time(yes, no)?')
# plot_live_graph_input = str(input())
# if(plot_live_graph_input is yes):
#     plot_live_graph = True
# else:
#     plot_live_graph= False

save_video_filepath='/Users/apple/OneDrive - Imperial College London/Simulation/amm.mp4'
plot_live_graph = False

# set pricing function to use
pricing_function_parameters = pricingFunctions.linearPricingFunction() #We may wish to have different pricing functions in future


# Below outputs code profiling describing performance of code
for flex_level_key, requests in requests_flex_levels_dict.items():

        pr = cProfile.Profile()
        pr.enable()

        my_result = energyMarket.allocateEnergy(execution_type_parameters, execution_times, neighbours_list, requests, neighbours_consumption_ts_list, consumption_prediction, supply_input, pricing_function_parameters, scaling_methodology, flex_level_key)

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()

        with open('test.txt', 'w+') as f:
                f.write(s.getvalue())


print('We are finished')

###################################################################