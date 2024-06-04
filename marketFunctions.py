# Author: Shaun Sweeney 
# Contact: shaunsweeney12@gmail.com
# Initial creation: June 2023

# Purpose: 
# This script contains an extensive set of functions used mostly by marketFunctions.py including core functions relating to energy allocation methodologies and pricing


import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import pandas as pd
import numpy as np
import time
import random
import pyomo.environ as pyo
import marketDesignFunctions as mdf
import datetime as dt
import sys
from collections import defaultdict
from copy import deepcopy



num_seconds_hour=60*60

def oldCalculateGridHealth(current_supply, compet_supply, initial_supply):
    grid_health_function = pd.Series(np.zeros(len(current_supply)), index = current_supply.index)
    max_grid_health=100
    min_grid_health=0
    max_available_flexible_supply = max(initial_supply)
    min_available_flexible_supply = 0 # min(initial_supply)

    for time_step, time_step_supply in enumerate(current_supply):
        if(current_supply[time_step] > compet_supply[time_step]): 
            grid_health_function[time_step] = 100
        elif(compet_supply[time_step]==0):
            grid_health_function[time_step] = 0
        else:
            grid_health_function[time_step] = (time_step_supply/(max_available_flexible_supply-min_available_flexible_supply))*(max_grid_health-min_grid_health)
    return grid_health_function 


def calculateGridHealth(consumption, supply):

    supply.name = 'Total supply'
    new_df = consumption.merge(supply.to_frame(), left_index=True, right_index=True)

    new_df['supply_ratio'] = new_df['Total supply'].divide(new_df['Total consumption'])

    new_df.loc[new_df['supply_ratio'] >= 1, 'grid_health'] = 100
    new_df.loc[new_df['supply_ratio'] < 1, 'grid_health'] = new_df['supply_ratio']*100

    return new_df['grid_health']



def addRowToDebuggingCSV(rows, requests, market_iter, market_start_time, market_end_time, keyword, neighbours_list, supply_available_within_constraints,
                ts_grid_health, min_price_time):
    
    num_seconds_hour = 60*60

    for request in requests:
        neighbour_id = request.neighbour_id
        neighbour_parameters_only = extract(neighbours_list)
        neighbour = next((x for x in neighbour_parameters_only if x.id == neighbour_id), None)
        neighbour_position = neighbour_parameters_only.index(neighbour)


        if(type(ts_grid_health) == pd.core.series.Series):
            marketAverageGridHealth = ts_grid_health.loc[request.earliest_start_time:request.latest_end_time].mean()  
        else:
            marketAverageGridHealth=None


        if(len(neighbour.fairness_time_series)==0):
            fairness_for_csv_last=None
            fairness_for_csv = None
        else: 
            fairness_for_csv_last = neighbour.fairness_time_series[-1]
            fairness_for_csv = neighbour.fairness_time_series


        if(keyword !="queue"):
            try_satisfy=None
        else:
            try_satisfy = request.ever_tried_to_satisfy_request

                   
        new_row = pd.DataFrame({"Neighbour ID":[request.neighbour_id],
                        "Appliance ID":[request.appliance_id],
                        "Market Iteration":[market_iter],
                        "Request Iteration":[request.iter],
                        "Request ID":[request.id],
                        "Log Location":[keyword],
                        "Queue Position":[request.queue_position],
                        "Request Fairness":[request.fairness],
                        "Market Start Time":[market_start_time],
                        "Market End Time":[market_end_time],
                        "Earliest Start Time":[request.earliest_start_time],
                        "Beginning of Latest Running Period":[request.latest_end_time], #Request latest end time corresponds with the beginning of the period in which it last can run in
                        "Request Power":[request.power],
                        "Request Duration":[request.duration],
                        # "Market Energy Available":[supply_available_within_constraints],
                        # "Market Average Grid Health":[marketAverageGridHealth],
                        # "Min Price Time":[min_price_time],
                        "Ever Tried Satisfy Request":[try_satisfy],
                        "Request Acceptance":[request.accepted],
                        "Accepted Start Time":[request.accepted_start_time],
                        "Accepted End Time":[request.accepted_end_time],
                        "Request Cost":[request.cost],
                        "Neighbour Fairness":[fairness_for_csv_last],
                        "Neighbour Total Energy Desired":[neighbour.total_historic_energy_desired],
                        "Neighbour Total Energy Delivered":[neighbour.total_historic_energy_delivered],
                        "Request Total Energy Desired":[request.total_energy_desired],
                        "Request Total Energy Delivered":[request.total_energy_delivered],
                        "Request Type":[request.product],
                        "Processed Time":[time.time()],
                        "Neighbour Fairness Series":[fairness_for_csv]
                        })
        
        rows.append(new_row)

    return rows
    

def getInitialPowerAndCostValues(essential_consumption_type, execution_parameters, dt_time_series, neighbours_list, neighbours_consumption_ts_list, supply_input, consumption_prediction, time_series, ts_time_series, pricing_function_parameters):
    total_neighbours=len(neighbours_list)
    acceptable_num_households=int(0.7*total_neighbours) #if we have at least 70% of the data


    total_baseload_consumption=pd.DataFrame(np.zeros(execution_parameters.get("num_execution_samples")), index = dt_time_series, columns=['Baseload consumption'])

    for neighbour in neighbours_list:
        if(essential_consumption_type=="fixed"):
            total_baseload_consumption['Baseload consumption'] += neighbour[0].neighbour_baseload
        elif(essential_consumption_type=="timeseries"):
            for current_neighbour_consumption_id, current_neighbour_consumption in enumerate(neighbours_consumption_ts_list):
                total_baseload_consumption['consumption{}'.format(current_neighbour_consumption_id)] = current_neighbour_consumption[1]['essential'] 
                total_baseload_consumption['num missing vals'] = total_baseload_consumption.isnull().sum(axis=1)
                total_baseload_consumption['Baseload consumption'] = np.where(total_baseload_consumption['num missing vals']>(total_neighbours - acceptable_num_households), np.nan, (total_baseload_consumption.sum(axis=1)-total_baseload_consumption['num missing vals'])*(1+total_baseload_consumption['num missing vals']/total_neighbours))
                total_baseload_consumption['Baseload consumption'].fillna(method='ffill', inplace=True)
        elif(essential_consumption_type=="none"):
            total_baseload_consumption['Baseload consumption'] = 0


    # some checks done on missing data here
    supply_input = supply_input.tz_localize(None)
    initial_available_flexible_supply = supply_input - total_baseload_consumption['Baseload consumption'] 
    
    #store these values as they are required for output analysis
    initial_power_and_cost_values=pd.DataFrame(index = dt_time_series)
    initial_power_and_cost_values = initial_power_and_cost_values.merge(total_baseload_consumption['Baseload consumption'].to_frame(), left_index=True, right_index=True)
    initial_power_and_cost_values = initial_power_and_cost_values.merge(supply_input.to_frame(), left_index=True, right_index=True)
    initial_power_and_cost_values['flexible supply'] = initial_available_flexible_supply
    initial_power_and_cost_values['consumption_prediction'] = consumption_prediction
    initial_power_and_cost_values['grid_health'] = calculateGridHealth(consumption=consumption_prediction, supply=initial_available_flexible_supply).to_frame()
    initial_power_and_cost_values['buy_price'] = calculateMarketBuyPrice(initial_power_and_cost_values['grid_health'], pricing_function_parameters).to_frame() #this corresponds to the price for essential consumption
    initial_power_and_cost_values['sell_price'] = calculateMarketSellPrice(initial_power_and_cost_values['grid_health'], pricing_function_parameters).to_frame()
    
    #this outputs a file of starting values before allocation happens
    initial_power_and_cost_values.to_csv('files/initialPowerAndCostValues.csv')

    return initial_power_and_cost_values 



def calculateMarketBuyPrice(ts_grid_health, pricing_function_parameters):
     return round(pricing_function_parameters.get('buy_price_slope')*ts_grid_health + pricing_function_parameters.get('buy_price_y_intercept'), 2)


def calculateMarketSellPrice(ts_grid_health, pricing_function_parameters):
     return round(pricing_function_parameters.get('sell_price_slope')*ts_grid_health + pricing_function_parameters.get('sell_price_y_intercept'), 2)


def getExecutionParameters(execution_days, market_timings):
    num_seconds_day = 24*60*60 
    num_seconds_hour = 60*60

    num_daily_samples = num_seconds_day/market_timings.get("market_resolution").seconds
    num_execution_samples = int(num_daily_samples*execution_days)


    execution_parameters =  {"num_daily_samples": num_daily_samples, 
                             "num_execution_samples": num_execution_samples }
    
    return execution_parameters


def getExecutionParametersNew(start_time, end_time, market_parameters):
    num_seconds_day = 24*60*60 
    num_seconds_hour = 60*60

    num_daily_samples = num_seconds_day/market_parameters.get("market_resolution").seconds
    num_execution_samples = int((end_time-start_time)/market_parameters.get("market_resolution"))


    execution_parameters =  {"num_daily_samples": num_daily_samples, 
                             "num_execution_samples": num_execution_samples }
    
    return execution_parameters

def getMinPriceParameters(ts_energy_buy_price, request, market_timings):
    booking_earliest_start_time = max(request.earliest_start_time, market_timings.get("market_start_time"))
    booking_latest_end_time = min(request.latest_end_time, market_timings.get("market_end_time"))

    min_price = ts_energy_buy_price.loc[booking_earliest_start_time:booking_latest_end_time].min()          
    min_price_time = ts_energy_buy_price.loc[booking_earliest_start_time:booking_latest_end_time].idxmin()


    min_price_parameters = {"min_price": min_price,
                        "min_price_time": min_price_time}
    return min_price_parameters



def getBookingParameters(request, market_timings, market_parameters, min_price_parameters):
    if(min_price_parameters.get("min_price_time") + (request.duration/2) > market_timings.get("market_end_time")):
        booking_end_time=market_timings.get("market_end_time")
        booking_start_time=booking_end_time-request.duration
    elif(min_price_parameters.get("min_price_time") - (request.duration/2) < market_timings.get("market_start_time")): #covers the case where the bisection below would cause the booking start time to be set sooner than the market start time
        booking_start_time = market_timings.get("market_start_time")
        booking_end_time = booking_start_time + request.duration 
    else:
        booking_start_time=min_price_parameters.get("min_price_time")-(request.duration/2)
        booking_end_time = booking_start_time + request.duration 

    booking_parameters={"booking_start_time": booking_start_time,
                        "booking_end_time": booking_end_time}

    return booking_parameters


def updateRequestStatus(request, booking_parameters):
    num_seconds_hour=60*60
    
    request.accepted_start_time=booking_parameters.get("booking_start_time")
    request.accepted_end_time=booking_parameters.get("booking_end_time")
    request.accepted=True

    return request


def updateRequestCost(request, ts_energy_buy_price):

    if(request.accepted==True):
        request.cost=round(request.total_energy_delivered*ts_energy_buy_price.loc[request.accepted_start_time: request.accepted_end_time].sum(), 2)
    else:
        request.cost = None

    return request


def setMarketTimings(market_parameters): 
    market_start_time = market_parameters.get("market_start_time")
    market_end_time = market_parameters.get("market_end_time")

    market_times =  {"market_start_time": market_start_time, 
                "market_end_time": market_end_time}
    
    return market_times


def updateRollingMarketTimings(market_timings, market_parameters):
    market_start_time = market_timings.get("market_start_time") + market_parameters.get("market_interval")
    market_end_time = market_start_time + market_parameters.get("market_window_length") - market_parameters.get("market_resolution") 

    market_times =  {"market_start_time": market_start_time, 
                    "market_end_time": market_end_time}
    
    return market_times

def updateOptimisationTimings(market_timings, market_parameters):
    market_start_time = market_timings.get("market_start_time") + market_parameters.get("optimisation_window_length")
    market_end_time = market_start_time + market_parameters.get("optimisation_window_length") - market_parameters.get("market_resolution") 

    market_times =  {"market_start_time": market_start_time, 
                    "market_end_time": market_end_time}
     
    return market_times


def initialiseDebuggingDataFrame():
    debugging_cols=['Neighbour ID', 'Appliance ID', 'Market Iteration', 'Request Iteration', 'Request ID', 'Log Location', 'Queue Position', 'Request Fairness', 
                    'Market Start Time', 'Market End Time', 'Earliest Start Time', 'Latest End Time', 'Request Power', 'Request Duration', 'Ever Tried Satisfy Request',
                    'Request Acceptance', 'Accepted Start Time', 'Accepted End Time', 'Request Cost', 'Neighbour Fairness','Neighbour Total Energy Desired', 'Neighbour Total Energy Delivered',
                    'Request Total Energy Desired', 'Request Total Energy Delivered', 'Request Type', 'Processed Time', 'Neighbour Fairness Series']
    debugging_df = pd.DataFrame(columns=debugging_cols)
    debugging_df = debugging_df.astype({'Request Acceptance': 'bool'})

    return debugging_df


def orderRequestQueue(neighbours_list, requests, market_timings, market_parameters):
    neartime_request_queue, longer_time_request_queue = splitRequestQueue(requests, market_timings, market_parameters)

    #order the queue by fairness (if using deterministic approach - non-stochastic)
    # requests_to_process_now_queue = orderQueueByFairness(neartime_request_queue, longer_time_request_queue)

    #order the queue randomly
    requests_to_process_now_queue = orderQueueRandomly(neartime_request_queue, longer_time_request_queue)

    return requests_to_process_now_queue
    


def splitRequestQueue(requests_to_process_queue, market_timings, market_parameters):
    neartime_request_queue = []
    longer_time_request_queue =[] 
      
    for request in requests_to_process_queue:
        # print('iter: '+ str(iter))
        if(request.earliest_start_time.tz != None):
            request.earliest_start_time=request.earliest_start_time.tz_convert(None)
        if(request.latest_end_time.tz != None):
            request.latest_end_time=request.latest_end_time.tz_convert(None)
        
        #create two queues - one for very short term requests (to maximise utilisation of energy) and the other for slightly term requests
        if(request.latest_end_time>request.earliest_start_time):
            if(request.latest_end_time <= (market_timings.get("market_start_time") + market_parameters.get("neartime_energy_window"))
            and request.latest_end_time >= (market_timings.get("market_start_time"))):
                neartime_request_queue.append(request)  
                request.iter+=1 #update iter to keep track of how many times the request has gone through the market
            elif((request.earliest_start_time <= market_timings.get("market_start_time") and 
            request.latest_end_time >= market_timings.get("market_start_time"))
            or 
            (request.earliest_start_time <= market_timings.get("market_end_time") and 
            request.latest_end_time >= market_timings.get("market_end_time"))
            or
            (request.earliest_start_time >= market_timings.get("market_start_time") and 
            request.latest_end_time <= market_timings.get("market_end_time"))):
                request.iter+=1 #update iter to keep track of how many times the request has gone through the market
                longer_time_request_queue.append(request)  
            elif(request.earliest_start_time > market_timings.get("market_end_time")):
                continue;
    return neartime_request_queue, longer_time_request_queue 

    
def determineIfRequestIsFeasible(available_flexible_supply, request, market_parameters):
    num_seconds_hour=60*60

    count=0
    for index, row in available_flexible_supply.loc[request.earliest_start_time:request.latest_end_time].items():
        feasible = False
        if row >= request.power:
            count +=1
        else: 
            count=0
        
        if count >= request.duration/market_parameters.get("market_resolution"):

            feasible = True
            break

    return feasible

    # supply_available_within_constraints = sum(available_flexible_supply.loc[request.earliest_start_time:request.latest_end_time])*market_parameters.get("market_resolution").seconds/num_seconds_hour
    # return supply_available_within_constraints


def extract(lst):
    return [item[0] for item in lst]
     

#update the value of the fairness metric
def updateFairnessMetrics(execution_type_parameters, neighbours_list, requests):

    for request in requests:
        neighbour_id = request.neighbour_id
        neighbour_parameters_only = extract(neighbours_list)
        neighbour = next((x for x in neighbour_parameters_only if x.id == neighbour_id), None)
        neighbour_position = neighbour_parameters_only.index(neighbour)
        

        # below is for updating energy desired, delivered and fairness scores when using the queuing methodology (or considering fairness)
        # this is a necesssarily different approach to the global optimisation approaches (volume or revenue maximising) due to
        if(execution_type_parameters.get("resource_allocation_method") == "queue"): 

            #update energy desired       
            request.total_energy_desired = request.power*request.duration.seconds/num_seconds_hour
            
            # there needs to be some additional checks due to the stochasticity in the system chanigng how requests are updated
            if(request.updated_energy_desired==False):
                if(execution_type_parameters.get("appliances_labelled")==True):
                    neighbours_list[(request.appliance_id)].total_historic_energy_desired += request.total_energy_desired
                elif(execution_type_parameters.get("appliances_labelled")==False):
                    neighbour.total_historic_energy_desired += request.total_energy_desired
                    request.updated_energy_desired = True

            #update energy delivered
            if(request.accepted==True):
                request.total_energy_delivered = request.power*request.duration.seconds/num_seconds_hour
                if(execution_type_parameters.get("appliances_labelled")==True):
                    neighbour[(request.appliance_id)].total_historic_energy_delivered += request.total_energy_delivered
                elif(execution_type_parameters.get("appliances_labelled")==False):
                    neighbour.total_historic_energy_delivered += request.total_energy_delivered
            else:
                 request.total_energy_delivered = 0


            #update fairness score
            if(execution_type_parameters.get("appliances_labelled")==True):
                request.fairness = request.total_energy_delivered/request.total_energy_desired
                neighbourfairness = round(neighbour[(request.appliance_id)].total_historic_energy_delivered/neighbour[(request.appliance_id)].total_historic_energy_desired, 2)
                neighbour[(request.appliance_id)].fairness_time_series.append(neighbourfairness)
            elif(execution_type_parameters.get("appliances_labelled")==False):
                    request.fairness = request.total_energy_delivered/request.total_energy_desired
                    neighbourfairness = round(neighbour.total_historic_energy_delivered/neighbour.total_historic_energy_desired, 2)
                    neighbour.fairness_time_series.append(neighbourfairness)


        # below is for updating energy desired, delivered and fairness scores when using the volume or revenue maximising stategies
        elif((execution_type_parameters.get("resource_allocation_method") == "volumeoptimisation") or (execution_type_parameters.get("resource_allocation_method") == "priceoptimisation")):
            if(execution_type_parameters.get("appliances_labelled")==True):
                neighbour[(request.appliance_id)].total_historic_energy_desired += request.total_energy_desired
                neighbour[(request.appliance_id)].total_historic_energy_delivered += request.total_energy_delivered
                fairness = neighbour[(request.appliance_id)].total_historic_energy_delivered/neighbour[(request.appliance_id)].total_historic_energy_desired
                neighbour[(request.appliance_id)].fairness_time_series.append(fairness)
            elif(execution_type_parameters.get("appliances_labelled")==False):
                
                neighbour.total_historic_energy_desired += request.total_energy_desired
                neighbour.total_historic_energy_delivered += request.total_energy_delivered
                fairness = neighbour.total_historic_energy_delivered/neighbour.total_historic_energy_desired
                neighbour.fairness_time_series.append(fairness)

        neighbours_list[neighbour_position][0] = neighbour

    return requests, neighbours_list





def getRequestAndNeighbourFairness(execution_type_parameters, request, neighbour):
    if(execution_type_parameters.get("appliances_labelled")==True):
        neighbour_energy_delivered = neighbour[request.appliance_id].total_historic_energy_delivered
        neighbour_energy_desired = neighbour[request.appliance_id].total_historic_energy_desired
        neighbour_fairness=None

        if(len(neighbour[request.appliance_id].fairness_time_series)>=0):
            if(neighbour_energy_desired>0):
                request.fairness = neighbour_energy_delivered /neighbour_energy_desired
                neighbour_fairness = neighbour[request.appliance_id].fairness_time_series[-1]
    elif(execution_type_parameters.get("appliances_labelled")==False):
        request.fairness =  request.total_energy_delivered/request.total_energy_desired

        neighbour_energy_delivered = neighbour[0].total_historic_energy_delivered
        neighbour_energy_desired = neighbour[0].total_historic_energy_desired
        neighbour_fairness=None

        if(len(neighbour[0].fairness_time_series)>=0):
            if(neighbour_energy_desired>0):
                neighbour_fairness = neighbour[0].fairness_time_series[-1]

    return request.fairness, neighbour_fairness, neighbour_energy_desired, neighbour_energy_delivered


def orderQueueByFairness(neartime_request_queue, longer_time_request_queue):
    neartime_request_queue.sort(key=lambda request: request.fairness, reverse=False)
    longer_time_request_queue.sort(key=lambda request: request.fairness, reverse=False)
    requests_to_process_now_queue = neartime_request_queue + longer_time_request_queue

    return requests_to_process_now_queue



def identifyReleventRequests(requests, market_timings):
    requests_to_process_now=[]
    for request in requests:  

        if((request.earliest_start_time <= market_timings.get("market_start_time") and 
            ((request.latest_end_time - request.duration) >= market_timings.get("market_start_time"))
            or 
            ((request.earliest_start_time + request.duration)) <= market_timings.get("market_end_time") and 
            request.latest_end_time >= market_timings.get("market_end_time"))
            or
            (request.earliest_start_time >= market_timings.get("market_start_time") and 
            request.latest_end_time <= market_timings.get("market_end_time"))):
            if(request.accepted != True):
                requests_to_process_now.append(request)

        elif(request.earliest_start_time > market_timings.get("market_end_time")):
            break
        else:
            continue
    
    return requests_to_process_now




def orderNeartimeAndLongerTimeQueuesRandomly(neartime_request_queue, longer_time_request_queue):
    random.shuffle(neartime_request_queue)
    random.shuffle(longer_time_request_queue)
    requests_to_process_now_queue = neartime_request_queue + longer_time_request_queue

    return requests_to_process_now_queue


def orderQueueRandomly(queue):
    random.shuffle(queue)
    return queue


def getNumberOfEachContractType(neighbours_list):

    neighbour_contracts = []

    for neighbour_id, neighbour in enumerate(neighbours_list):
        neighbour_contracts.append(neighbours_list[neighbour_id][0].product)
        
    number_basic_contracts = neighbour_contracts.count('basic')
    number_premium_contracts = neighbour_contracts.count('premium')

    number_contracts = {"number_basic_contracts":number_basic_contracts,
                        "number_premium_contracts":number_premium_contracts}
    
    return number_contracts


def identifyPremiumAndBasicRequests(requests):

    premium_requests_to_process_now=[]
    basic_requests_to_process_now=[]

    for request in requests:
        if request.product == "basic":
            basic_requests_to_process_now.append(request)
        elif request.product == "premium":
            premium_requests_to_process_now.append(request)

    return premium_requests_to_process_now, basic_requests_to_process_now


def roundTime(date, delta):
    return date + (dt.min - date) % delta


def convertPdTimeIndexToUnix(timeSeries, market_parameters):
    tsWithUnixIndex = timeSeries.copy()
    tsWithUnixIndex.index = ((tsWithUnixIndex.index- pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))// market_parameters.get("market_resolution").total_seconds() 
    tsWithUnixIndex.index = tsWithUnixIndex.index.astype('int64')

    return tsWithUnixIndex

def allocateProbabilitySuccessRequest(requestList, neighboursList):

    total_request_fairness = 0
    for request in requestList:

        neighbour_id = request.neighbour_id
        neighbour_parameters_only = extract(neighboursList)
        neighbour = next((x for x in neighbour_parameters_only if x.id == neighbour_id), None)
        neighbour_position = neighbour_parameters_only.index(neighbour)

        if(len(neighbour.fairness_time_series)>0):
            total_request_fairness+= neighbour.fairness_time_series[-1]
        else:
            total_request_fairness +=1 #if no requests have been submitted yet, use 100% as initial conditon

    #normalise probability scores based on all requests in queue in line with stochastic resource allocation methodology
    temp_increase_success_factor = 1
    for request in requestList:
        if(len(neighbour.fairness_time_series)>0):
            request.temp_prob_success = temp_increase_success_factor*neighbour.fairness_time_series[-1]/total_request_fairness
        else:
            request.temp_prob_success = temp_increase_success_factor/total_request_fairness

    return requestList


def allocateOneProbabilitySuccessRequest(requestList):

    for request in requestList:
        request.temp_prob_success = 1

    return requestList


def identifyRandomRequestFromQueue(requestList):
    if(len(requestList)!=0):
        request_id = random.randint(0, len(requestList)-1)         
        request = requestList[request_id]
        return request
    else:
        return None


#biased coin toss methodology
def flipBiasedCoin(biased_prob_success):
    return True if random.random() > biased_prob_success else False

 

def invert_curve(curve):
    #our optimisation is required to have a maximising objective function - so we must invert the price curve 
    new_curve = -1*curve + max(curve) + 1
    return new_curve


# this is the main function used for processing requests according to the stochastic queuing methodology which considers fairness
def allocateEnergyAccordingToQueues(execution_type_parameters, execution_times, execution_parameters, market_parameters, pricing_function_parameters, market_timings, market_iter, dt_time_series, initial_power_and_cost_values, consumption_prediction, requests, neighbours_list, ts_available_flexible_supply, ts_grid_health, ts_energy_buy_price, ts_energy_sell_price, booked_flexible_consumption, scaling_methodology, flex_level_key):
    try_to_satisfy_request=False
    rows=[]

    if(execution_type_parameters.get("repeat_same_day_multiple_times")==True and market_iter==0):
        ts_available_flexible_supply_init = pd.Series(deepcopy(ts_available_flexible_supply.to_dict()))
        booked_flexible_consumption_init = pd.Series(deepcopy(booked_flexible_consumption.to_dict()))
        ts_grid_health_init = pd.Series(deepcopy(ts_grid_health.to_dict()))


    if(market_iter==0):

        if(execution_type_parameters.get("use_products")=="yes"):
            basic_product_weight = mdf.getBasicProductInfo().get("basic_product_weight")
            premium_produuct_weight = mdf.getPremiumProductInfo().get("premium_product_weight")
            deadband_percentage = mdf.getDeadbandPercentage()

            number_contracts = getNumberOfEachContractType(neighbours_list)
            number_basic_contracts = number_contracts.get("number_basic_contracts")
            number_premium_contracts = number_contracts.get("number_premium_contracts")

            total_weight_basic_contracts = basic_product_weight*number_basic_contracts
            total_weight_premium_contracts = premium_produuct_weight*number_premium_contracts

            ts_premium_supply = total_weight_premium_contracts/(total_weight_basic_contracts+total_weight_premium_contracts)*(ts_available_flexible_supply*(1-deadband_percentage))
            ts_basic_supply = total_weight_basic_contracts/(total_weight_basic_contracts+total_weight_premium_contracts)*(ts_available_flexible_supply*(1-deadband_percentage))
            # ts_deadband_supply = ts_available_flexible_supply - ts_premium_supply - ts_basic_supply

        if(execution_type_parameters.get("repeat_same_day_multiple_times")==True):
            current_number_of_daily_repeats=1


    while(market_timings.get("market_end_time") < execution_times.get("end_time")):     
        print(market_timings.get("market_end_time")) 
        market_iter+=1 
        requests_to_process_now_queue=[]

        if(execution_type_parameters.get("repeat_same_day_multiple_times")==True):
            ts_available_flexible_supply = pd.Series(deepcopy(ts_available_flexible_supply_init.to_dict()))
            booked_flexible_consumption = pd.Series(deepcopy(booked_flexible_consumption_init.to_dict()))
            ts_grid_health = pd.Series(deepcopy(ts_grid_health_init.to_dict()))
            ts_energy_buy_price = calculateMarketBuyPrice(ts_grid_health, pricing_function_parameters) 
            ts_energy_sell_price = calculateMarketSellPrice(ts_grid_health, pricing_function_parameters) 
      

        first_request=True

        if(execution_type_parameters.get("repeat_same_day_multiple_times")==True and market_iter == 1):
            requests_to_process_now_queue = identifyReleventRequests(requests, market_timings)
        elif(execution_type_parameters.get("repeat_same_day_multiple_times")==True and market_iter != 1):
            requests_to_process_now_queue = requests_to_process_now_daily_repeat_init
            requests_to_process_now_daily_repeat = requests_to_process_now_daily_repeat_init
        else:
            requests_to_process_now_queue = identifyReleventRequests(requests, market_timings)
            
        requests_to_process_now_queue = allocateProbabilitySuccessRequest(requests_to_process_now_queue, neighbours_list)
  
        max_number_request_attempts= 200*len(requests_to_process_now_queue)
        request_processing_attempts=0

        if(execution_type_parameters.get("use_products")=="yes"):
            premium_requests_to_process_now, basic_requests_to_process_now = identifyPremiumAndBasicRequests(requests_to_process_now_queue)
            premium_requests_to_process_now = orderQueueRandomly(premium_requests_to_process_now)
            basic_requests_to_process_now = orderQueueRandomly(basic_requests_to_process_now)
            requests_to_process_now_queue = premium_requests_to_process_now + basic_requests_to_process_now

            premium_dict = {"name": "premium",
                            "queue": premium_requests_to_process_now,
                            "supply": ts_premium_supply.loc[market_timings.get("market_start_time"):market_timings.get("market_end_time")]}
            basic_dict = {"name": "basic",
                        "queue":basic_requests_to_process_now,
                        "supply": ts_basic_supply.loc[market_timings.get("market_start_time"):market_timings.get("market_end_time")]}
            deadband_dict = {"name": "deadband",
                            "queue": requests_to_process_now_queue,
                            "supply": ts_available_flexible_supply.loc[market_timings.get("market_start_time"):market_timings.get("market_end_time")]}

            queues = [premium_dict, basic_dict, deadband_dict]

        elif(execution_type_parameters.get("use_products")=="no"):
            requests_to_process_now_queue = orderQueueRandomly(requests_to_process_now_queue)

            single_product_dict = {"name": "single_product",
                        "queue": requests_to_process_now_queue,
                        "supply": ts_available_flexible_supply.loc[market_timings.get("market_start_time"):market_timings.get("market_end_time")]}
            
            queues = [single_product_dict]

        for queue in queues:
            
            if(queue.get("name")=="deadband"):
                requests_to_process_now_queue = orderQueueRandomly(requests_to_process_now_queue)
                queue.update({"name": "deadband",
                        "queue": requests_to_process_now_queue,
                        "supply": ts_available_flexible_supply.loc[market_timings.get("market_start_time"):market_timings.get("market_end_time")]}) #this is to make sure the final queue has the most up to date values

            if(execution_type_parameters.get("repeat_same_day_multiple_times")==True and market_iter==1):
                requests_to_process_now_daily_repeat_init = requests_to_process_now_queue.copy()
                requests_to_process_now_daily_repeat = requests_to_process_now_daily_repeat_init.copy()
            elif(execution_type_parameters.get("repeat_same_day_multiple_times")==True and market_iter!=1):
                requests_to_process_now_daily_repeat = requests_to_process_now_daily_repeat_init.copy()
                for request in requests_to_process_now_daily_repeat:
                    request.accepted = False
                    request.feasible_to_satisfy_request_this_iter = None
                    request.updated_energy_desired=False
                    

            if(len(queue.get("queue"))>0):

                if(execution_type_parameters.get("repeat_same_day_multiple_times")==False):
                    feasibleQueue=queue.get("queue") 
                else:
                    feasibleQueue = requests_to_process_now_daily_repeat_init
                while((request_processing_attempts < max_number_request_attempts) and len(feasibleQueue)>0):
                   

                    if(execution_type_parameters.get("repeat_same_day_multiple_times")==True):
                        feasibleQueue = [x for x in requests_to_process_now_daily_repeat  if (x.accepted !=True and x.feasible_to_satisfy_request_this_iter !=False)]
                    else:
                        feasibleQueue = [x for x in queue.get("queue") if (x.accepted !=True and x.feasible_to_satisfy_request_this_iter !=False)]


                    if(execution_type_parameters.get("repeat_same_day_multiple_times")==True):
                        for request in feasibleQueue:
                            request.accepted=False


                    print("length feasible queue:" +str(len(feasibleQueue)))
                    
                    request = identifyRandomRequestFromQueue(feasibleQueue)
                    if(request==None):
                        break;
                

                    #do not try to satisfy the request again if it is already accepted, ideally we would remove it from the queue once accepted but this presents some challenges for exporting the data
                    if(request.accepted!=True):
                        
                        request.iter+=1
                        request_index = request_processing_attempts
                        request.queue_position = request_index #debugging

                
                        try_to_satisfy_request = flipBiasedCoin(request.temp_prob_success) 
                        request.try_to_satisfy_request = try_to_satisfy_request
                        if(try_to_satisfy_request==True):
                            request.ever_tried_to_satisfy_request=True

                        if(try_to_satisfy_request == True):
                            
                            # supply_for_request = queue.get("supply").loc[request_params_list.get("request_time_constraints_dict")[0]:request_params_list.get("request_time_constraints_dict")[1]]
                            supply_for_request = queue.get("supply").loc[request.earliest_start_time:request.latest_end_time]
                            supply_unix = convertPdTimeIndexToUnix(queue.get("supply"), market_parameters)
                            supply_for_request_unix = convertPdTimeIndexToUnix(supply_for_request, market_parameters)
                            request_params_list = getRequestParamsForOptimisation([request], supply_for_request_unix, market_parameters)

                            #adding rounding below as solver ignores this on each iteration which has caused issues
                            ts_energy_buy_price_unix = round(convertPdTimeIndexToUnix(ts_energy_buy_price, market_parameters), 2)
                            ts_energy_buy_price_unix_for_request = ts_energy_buy_price_unix.loc[request_params_list.get("request_time_constraints_dict")[0][0]:request_params_list.get("request_time_constraints_dict")[0][1]]

                            ts_energy_buy_price_unix_for_request_adapted = round(invert_curve(ts_energy_buy_price_unix_for_request), 2)
                            
                            num_periods = len(supply_for_request_unix)
                            eligible_starts = defineEligibleStarts(request_params_list, num_periods)

                            #create model to find the optimal slot for the request
                            m = pyo.ConcreteModel('dispatch')

                            #define sets
                            m.T = pyo.Set(initialize=tuple(supply_unix.index))
                            m.R = pyo.Set(initialize=tuple(request_params_list.get("request_max_power_dict").keys()))
                            m.windows = pyo.Set(m.T, initialize=eligible_starts, within=m.R, doc="requests eligible in each timeslot")
                            m.windows_flat = pyo.Set(initialize={(t, r) for t in eligible_starts for r in eligible_starts[t]},within=m.T * m.R)

                            ## PARAMS
                            m.request_power_size = pyo.Param(m.R, initialize=request_params_list.get("request_max_power_dict"))
                            m.request_energy_size = pyo.Param(m.R, initialize=request_params_list.get("request_max_energy_dict"))
                            m.power_limit = pyo.Param(m.T, initialize=supply_unix.to_dict())
                            m.duration_periods = pyo.Param(m.R, initialize=request_params_list.get("request_duration_num_periods_dict"))
                            m.cost_of_energy =  pyo.Param(m.T, initialize=ts_energy_buy_price_unix_for_request_adapted.to_dict())
                        
                            ### VARS
                            m.dispatch = pyo.Var(m.windows_flat, domain=pyo.Binary, doc="dispatch power in timeslot t to request r")
                            
                            ## Objective - find the cheapest available slot for the individual
                            m.obj = pyo.Objective(expr=sum(m.dispatch[t, r]*m.request_energy_size[r]*m.cost_of_energy[t] for (t, r) in m.windows_flat), sense=pyo.maximize)


                            ### CONSTRAINTS
    
                            @m.Constraint(m.R)
                            def satisfy_only_once(m, r):
                                return sum(m.dispatch[t, rr] for (t, rr) in m.windows_flat if r == rr) <= 1

                            @m.Constraint(m.T)
                            def supply_limit(m, t):
                                # we need to sum across all dispatches that could be running in this period
                                possible_dispatches = {
                                    (tt, r) for (tt, r) in m.windows_flat if 0 <= t - tt < request_params_list.get("request_duration_num_periods_dict")[r]
                                }
                                if not possible_dispatches:
                                    return pyo.Constraint.Skip
                                return (
                                    sum(m.dispatch[tt, r] * m.request_power_size[r] for tt, r in possible_dispatches)
                                    <= m.power_limit[t]
                                )

                            solver = pyo.SolverFactory('cbc')
                            solver.options = { 'sec': 3600,  'threads':6} 

                            res = solver.solve(m)

                            # temp = sys.stdout 
                            # f = open('files/optlogs/output'+str(int(time.time()))+'_'+str(request.id).replace(":", "_")+'.txt', 'w')
                            # sys.stdout = f
                            # m.pprint()
                            # print(res)

                            # f.close()
                            # sys.stdout = temp

                
                            for r in m.R:
                                timeslots = {t for t in m.T if r in m.windows[t]}

                                for t in timeslots: 
                                    if(m.dispatch[t,r]()==1):
                                        request.accepted=True
                                        accepted_start_time_unix=t
                       
                                        accepted_end_time_unix = accepted_start_time_unix + int(request_params_list.get("request_duration_num_periods_dict")[r])
            
                                        request.accepted_start_time =  pd.to_datetime("1970-01-01") + pd.Timedelta(seconds = accepted_start_time_unix*market_parameters.get("market_resolution").total_seconds(), unit='ms')
                                        request.accepted_end_time =  pd.to_datetime("1970-01-01") + pd.Timedelta(seconds = accepted_end_time_unix*market_parameters.get("market_resolution").total_seconds(), unit='ms')
                                                                                
 
                            if(request.accepted==True):
                                queue.get("supply").loc[request.accepted_start_time: request.accepted_end_time-market_parameters.get("market_resolution")] -= request.power
                                queue.get("supply").loc[request.accepted_start_time: request.accepted_end_time-market_parameters.get("market_resolution")] = round(queue.get("supply").loc[request.accepted_start_time: request.accepted_end_time-market_parameters.get("market_resolution")], 2)
                                if(queue.get("name")!="deadband") & (queue.get("name")!="single_product"):
                                    ts_available_flexible_supply.loc[request.accepted_start_time: request.accepted_end_time-market_parameters.get("market_resolution")] -= request.power
                                booked_flexible_consumption.loc[request.accepted_start_time: request.accepted_end_time] += request.power

                            else: #request can't be met
                                request.accepted_start_time=None
                                request.accepted_end_time=None
                                request.accepted=False
                                request.feasible_to_satisfy_request_this_iter = False
                        else: #request can't be met
                            request.accepted_start_time=None
                            request.accepted_end_time=None
                            request.accepted=False

                        if(request.accepted==True):
                            request.total_energy_delivered = request.power*request.duration.seconds/num_seconds_hour
                            request = updateRequestCost(request, ts_energy_buy_price)
                            ts_grid_health.loc[request.accepted_start_time: request.accepted_end_time-market_parameters.get("market_resolution")] = calculateGridHealth(supply=ts_available_flexible_supply.loc[request.accepted_start_time: request.accepted_end_time-market_parameters.get("market_resolution")], consumption=consumption_prediction.loc[request.accepted_start_time: request.accepted_end_time-market_parameters.get("market_resolution")])
                            ts_energy_buy_price = calculateMarketBuyPrice(ts_grid_health, pricing_function_parameters) 
                            ts_energy_sell_price = calculateMarketSellPrice(ts_grid_health, pricing_function_parameters) 


                        request_processing_attempts +=1

                
        keyword = "queue"
        temp, neighbours_list = updateFairnessMetrics(execution_type_parameters, neighbours_list, queue.get("queue"))
        queue.update({"queue": temp})
        rows = addRowToDebuggingCSV(rows, queue.get("queue"), market_iter, market_timings.get('market_start_time'), market_timings.get('market_end_time'), keyword, neighbours_list, None,None, None)
        

        for request in queue.get("queue"):
            if(request.accepted==True):
                if(execution_type_parameters.get("repeat_same_day_multiple_times")==False):
                    queue.get("queue").remove(request)
                    if(execution_type_parameters.get("repeat_same_day_multiple_times")==False):
                        requests.remove(request) #global list of requests

                    if((queue.get("name")!="deadband") & (queue.get("name")!="single_product")):
                        requests_to_process_now_queue.remove(request) 

                elif(execution_type_parameters.get("repeat_same_day_multiple_times")==True):
                    requests_to_process_now_daily_repeat.remove(request)
                    
        if(execution_type_parameters.get("repeat_same_day_multiple_times")==False):
            market_timings = updateRollingMarketTimings(market_timings, market_parameters)
        else:
            current_number_of_daily_repeats+=1

    debugging_df = pd.concat(rows, ignore_index = True)
    debugging_df.to_csv('outputanalysis/debugging'+keyword+str(int(time.time()))+'_scale'+str(scaling_methodology.get("value"))+flex_level_key+'.csv')  
    
    #save supply remaining available at end of market instance - for debugging
    queue.get("supply").to_csv('supply'+keyword+str(int(time.time()))+'.csv')



# translate variables to the format they need to be for pyomo
def getRequestParamsForOptimisation(requests, supply, market_parameters):

    params_dict={}
    request_time_constraints_dict = {}
    request_max_power_dict = {}
    request_max_price_dict={}
    request_max_energy_dict={}
    sampling_period_dict={}
    request_duration_num_periods_dict = {}
    request_duration_hrs__dict={}
    request_latest_start_time_dict={}


    for request_iter, request in enumerate(requests):

        request_earliest_start_time = int(((request.earliest_start_time - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')) // market_parameters.get("market_resolution").total_seconds())
        request_latest_end_time = int(((request.latest_end_time  - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')) // market_parameters.get("market_resolution").total_seconds())
        request_num_periods = ((pd.to_datetime(pd.Timestamp("1970-01-01") + request.duration) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')) / (market_parameters.get("market_resolution").total_seconds())
        
        if(request_earliest_start_time < supply.index[0]):
            request_earliest_start_time = int(supply.index[0])

        if(request_latest_end_time > supply.index[-1]):
            request_latest_end_time = int(supply.index[-1])

        request_latest_start_time = request_latest_end_time - request_num_periods +1

        request_time_constraints_dict[request_iter] = (int(request_earliest_start_time), int(request_latest_start_time))
        request_max_power_dict[request_iter] = request.power
        request_max_energy_dict[request_iter] = round(request.power*request.duration.total_seconds()/num_seconds_hour,3)
        request_duration_num_periods_dict[request_iter] = request_num_periods
        request_max_price_dict[request_iter] = request.max_buy_price


    assert request_time_constraints_dict.keys() == request_max_power_dict.keys()
    assert request_time_constraints_dict.keys() == request_duration_num_periods_dict.keys()


    params_dict["request_duration_num_periods_dict"] = request_duration_num_periods_dict
    params_dict["request_max_energy_dict"] = request_max_energy_dict
    params_dict["request_max_power_dict"] = request_max_power_dict
    params_dict["request_time_constraints_dict"] = request_time_constraints_dict
    params_dict["request_max_price_dict"] = request_max_price_dict


    return params_dict


# function for optimisation in pyomo
def defineEligibleStarts(request_params_list, num_periods):
    eligible_starts = defaultdict(list)
    for r, (start, end) in request_params_list.get("request_time_constraints_dict").items():
        for t in [
            t for t in range(start, end+1) if t + request_params_list.get("request_duration_num_periods_dict")[r] <= (num_periods + t)
        ]:
            eligible_starts[t].append(r)

    return eligible_starts


# global optimisation - volume maximising or evenue maximising
def generalOptimisation(execution_type_parameters, execution_times, market_timings, market_parameters, supply, neighbours_list, requests, scaling_methodology, flex_level_key):

    market_iter=0
    rows=[]

    while(market_timings.get("market_end_time") < execution_times.get("end_time")): 
        print(market_timings.get("market_end_time")) 
        market_iter+=1 
        number_seconds_hour = 60*60
        sampling_period_hrs = market_parameters.get("market_resolution").total_seconds()/number_seconds_hour

        requests_to_process_now = identifyReleventRequests(requests, market_timings)

        supply_unix_ts = convertPdTimeIndexToUnix(supply.loc[market_timings.get("market_start_time"):market_timings.get("market_end_time")], market_parameters)

        supply_unix_ts = round(supply_unix_ts, 0)

        num_periods = len(supply_unix_ts.index)
        periods = tuple(range(supply_unix_ts.index.min(), supply_unix_ts.index.max()+1))

        request_ids=[]

        for request in requests_to_process_now:
            request_ids.append(request.id)
    
        # PARAMETERS
        request_params_list = getRequestParamsForOptimisation(requests_to_process_now, supply_unix_ts, market_parameters)

        eligible_starts = defineEligibleStarts(request_params_list, num_periods)

        
        ### MODEL BUILD

        m = pyo.ConcreteModel('dispatch')

        ### SETS
        m.T = pyo.Set(initialize=tuple(supply_unix_ts.index))
        m.R = pyo.Set(initialize=tuple(request_params_list.get("request_max_power_dict").keys()))
        # Note:  This will be an "indexed" set, where we have sets indexed by some index, in this case, m.T
        m.windows = pyo.Set(m.T, initialize=eligible_starts, within=m.R, doc="requests eligible in each timeslot")
        m.windows_flat = pyo.Set(initialize={(t, r) for t in eligible_starts for r in eligible_starts[t]},within=m.T * m.R)


        ## PARAMS
        m.request_power_size = pyo.Param(m.R, initialize=request_params_list.get("request_max_power_dict"))
        m.request_energy_size = pyo.Param(m.R, initialize=request_params_list.get("request_max_energy_dict"))
        m.power_limit = pyo.Param(m.T, initialize=supply_unix_ts.to_dict())
        m.duration_periods = pyo.Param(m.R, initialize=request_params_list.get("request_duration_num_periods_dict"))

      
        ### VARS
        m.booked_supply = pyo.Var(m.T, domain = pyo.NonNegativeReals)
        m.dispatch = pyo.Var(m.windows_flat, domain=pyo.Binary, doc="dispatch power in timeslot t to request r")


        ### OBJ
        if(execution_type_parameters.get("resource_allocation_method") == "volumeoptimisation"):
            keyword="volume"

            m.obj = pyo.Objective(expr=sum(m.dispatch[t,r]*m.request_energy_size[r] for (t, r) in m.windows_flat), sense=pyo.maximize)

        elif(execution_type_parameters.get("resource_allocation_method") == "priceoptimisation"):
            keyword="price"

            #additional parameter required for pricing optimisation
            m.request_max_buy_price_dict = pyo.Param(m.R, initialize = request_params_list.get("request_max_price_dict")) #parameters for max buy price


            # maximise revenue
            m.obj = pyo.Objective(expr=sum(m.dispatch[t, r]*m.request_energy_size[r]*m.request_max_buy_price_dict[r] for (t, r) in m.windows_flat), sense=pyo.maximize)



        ### CONSTRAINTS
            
        @m.Constraint(m.R)
        def satisfy_only_once(m, r):
            return sum(m.dispatch[t, rr] for (t, rr) in m.windows_flat if r == rr) <= 1


        @m.Constraint(m.T)
        def supply_limit(m, t):
            # we need to sum across all dispatches that could be running in this period
            possible_dispatches = {
                (tt, r) for (tt, r) in m.windows_flat if 0 <= t - tt < request_params_list.get("request_duration_num_periods_dict")[r]
            }
            if not possible_dispatches:
                return pyo.Constraint.Skip
            return (
                sum(m.dispatch[tt, r] * m.request_power_size[r] for tt, r in possible_dispatches)
                <= m.power_limit[t]
            )

        
        # solve it
        solver = pyo.SolverFactory('cbc')
        solver.options = { 'sec': 3600,  'threads':6} 
        # solver.options = { 'sec': 3600,  'threads':6,  'ratio': 0.01} 

        res = solver.solve(m)
        # self.solver = pyomo.opt.SolverFactory(SOLVER_NAME)
        # if 'cplex' in SOLVER_NAME:
        #     self.solver.options['timelimit'] = TIME_LIMIT
        # elif 'glpk' in SOLVER_NAME:         
        #     self.solver.options['tmlim'] = TIME_LIMIT
        # elif 'gurobi' in SOLVER_NAME:           
        #     self.solver.options['TimeLimit'] = TIME_LIMIT
        # elif 'xpress' in SOLVER_NAME:
        #     self.solver.options['soltimelimit'] = TIME_LIMIT 
        #     # Use the below instead for XPRESS versions before 9.0
        #     # self.solver.options['maxtime'] = TIME_LIMIT 


        # temp = sys.stdout 
        # f = open('files/output'+str(int(time.time()))+'.txt', 'w')
        # sys.stdout = f
        # m.pprint()
        # print(res)

        # f.close()
        # sys.stdout = temp


        assigned_periods = {}
        for t, r in m.dispatch:
            if pyo.value(m.dispatch[t, r]) > 0.5:  # request was assigned
                assigned_periods[r] = list(range(t, t + int(request_params_list.get("request_duration_num_periods_dict")[r])))
        total_period_power_assigned = []
        for t in m.T:
            assigned = 0
            for r in m.R:
                if t in assigned_periods.get(r, set()):
                    assigned += request_params_list.get("request_max_power_dict")[r]
            total_period_power_assigned.append(assigned)

  

        # plot graph of output
        plt.step(periods, [supply_unix_ts.to_dict()[p] for p in periods], color="g")
        assigned_periods = {}
        for t, r in m.dispatch:
            if pyo.value(m.dispatch[t, r]) > 0.5:  # request was assigned
                assigned_periods[r] = list(range(t, t + int(request_params_list.get("request_duration_num_periods_dict")[r])))
                # print("hit", t, r)
        total_period_power_assigned = []
        for t in m.T:
            assigned = 0
            for r in m.R:
                if t in assigned_periods.get(r, set()):
                    assigned += request_params_list.get("request_max_power_dict")[r]
            total_period_power_assigned.append(assigned)

        # print(total_period_power_assigned)

        plt.step(periods, total_period_power_assigned)
        plt.savefig('figs/'+keyword+'/'+str(int(time.time()))+'_'+str(supply_unix_ts.index.min())+'.png')
        plt.close()    
        
        rows, requests_to_process_now = translateModelOutputsToRequests(rows, execution_type_parameters, market_parameters, market_timings, neighbours_list, requests_to_process_now, m, market_iter, keyword, sampling_period_hrs, request_params_list.get("request_duration_num_periods_dict"))
        supply = updateGlobalSupply(m, supply, market_timings, market_parameters, total_period_power_assigned)
        market_timings = updateOptimisationTimings(market_timings, market_parameters)

    
    debugging_df = pd.concat(rows, ignore_index = True)
    debugging_df.to_csv('outputanalysis/debugging'+keyword+str(int(time.time()))+'_scale'+str(scaling_methodology.get("value"))+flex_level_key+'.csv')  
    supply.to_csv('supply'+keyword+str(int(time.time()))+'.csv')





def updateGlobalSupply(m, supply, market_timings, market_parameters, total_period_power_assigned):
    initially_available_supply_dict = m.power_limit.extract_values()
    

    supply_df = pd.DataFrame.from_dict(initially_available_supply_dict, orient='index', columns=['initially available supply'])   
    supply_df['booked supply'] = total_period_power_assigned
    supply_df['remaining_supply_available'] = supply_df['initially available supply'] - supply_df['booked supply']


    tsWithDtIndex = supply_df.copy()

    row_count=0
    new_index_vals=[]
    for index, row in tsWithDtIndex.iterrows():
        new_index_vals.append(pd.to_datetime("1970-01-01") + pd.Timedelta(seconds = tsWithDtIndex.index[row_count]*market_parameters.get("market_resolution").total_seconds(), unit='ms'))
        row_count+=1

    tsWithDtIndex['Time pandas'] = new_index_vals
    tsWithDtIndex.set_index("Time pandas", inplace=True)

    supply.loc[market_timings.get("market_start_time"):market_timings.get("market_end_time")] = tsWithDtIndex['remaining_supply_available']

    return supply


    
# back translate variables in format required for pyomo to standard format
def translateModelOutputsToRequests(rows, execution_type_parameters, market_parameters, market_timings, neighbours_list, requests, m, market_iter, keyword, sampling_period_hrs, request_duration_num_periods_dict):

    num_periods_market_res = num_seconds_hour/market_parameters.get("market_resolution").total_seconds()

    for r_idx, r in enumerate(m.R):
        requests[r_idx].total_energy_desired = m.request_energy_size[r_idx]
        timeslots = {t for t in m.T if r in m.windows[t]}

        energy_delivered=0
        request_accepted_temp = sum([m.dispatch[t,r]() for t in timeslots])
        if request_accepted_temp == 1:
            energy_delivered=requests[r_idx].total_energy_desired

        requests[r_idx].total_energy_delivered = energy_delivered
 
        if(requests[r_idx].total_energy_desired == requests[r_idx].total_energy_delivered): #do something to check whether we can consider request accepted to be true
            requests[r_idx].accepted = True

        #the below had been included as a post processing way to remove any partially fulfilled requests
        # elif(requests[r_idx].total_energy_delivered < requests[r_idx].total_energy_desired):
        #     requests[r_idx].total_energy_delivered == 0
    

        if(requests[r_idx].accepted == True):
            
            for t in timeslots: 
                if(m.dispatch[t,r_idx]()==1):
                    accepted_start_time_unix=t

            accepted_end_time_unix = accepted_start_time_unix + int(request_duration_num_periods_dict[r_idx])
  
            requests[r_idx].accepted_start_time =  pd.to_datetime("1970-01-01") + pd.Timedelta(seconds = accepted_start_time_unix*market_parameters.get("market_resolution").total_seconds(), unit='ms')
            requests[r_idx].accepted_end_time =  pd.to_datetime("1970-01-01") + pd.Timedelta(seconds = accepted_end_time_unix*market_parameters.get("market_resolution").total_seconds(), unit='ms')

        # for global optimisation - we assume the cost is equal to the maximum willingness to pay
        requests[r_idx].cost= requests[r_idx].max_buy_price*requests[r_idx].total_energy_delivered

    requests, neighbours_list = updateFairnessMetrics(execution_type_parameters, neighbours_list, requests)
    rows = addRowToDebuggingCSV(rows, requests, market_iter, market_timings.get('market_start_time'), market_timings.get('market_end_time'), keyword, neighbours_list, None,None, None)
        
    return rows, requests

