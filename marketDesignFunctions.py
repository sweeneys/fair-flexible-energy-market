
# Author: Shaun Sweeney 
# Contact: shaunsweeney12@gmail.com
# Initial creation: December 2022

# Purpose: 
# This script is where parameters relating to the market design can be set. 


import datetime as dt

def defaultMarketDesign(execution_times):
    neartime_energy_window = dt.timedelta(seconds=3*60*60) #prioritise requests for utilising energy in the next 3 hours
    market_interval = dt.timedelta(seconds=3*60*60) # a new market will start every 3 hours
    market_resolution = dt.timedelta(seconds=30*60) #30 minutes in seconds
    market_start_time = execution_times.get("start_time")
    market_window_length = dt.timedelta(seconds=24*60*60) #people can book energy up to 24 hours in advance
    market_end_time = market_start_time + market_window_length - market_resolution
    optimisation_window_length = dt.timedelta(seconds=24*60*60)


    market_design_parameters = dict({'neartime_energy_window': neartime_energy_window,
                    'market_interval': market_interval,
                    "market_start_time": market_start_time, 
                     "market_end_time": market_end_time,
                     "market_window_length": market_window_length,
                     "market_resolution": market_resolution,
                     "optimisation_window_length": optimisation_window_length})
    
    return market_design_parameters


def getDefaultMarketDesignParameters():
    neartime_energy_window = dt.timedelta(seconds=3*60*60) #prioritise requests for utilising energy in the next 3 hours
    market_interval = dt.timedelta(seconds=3*60*60) # a new market will start every 3 hours
    market_resolution = dt.timedelta(seconds=30*60) #30 minutes in seconds
    market_window_length = dt.timedelta(seconds=24*60*60) #people can book energy up to 24 hours in advance
    market_design_parameters = dict({'neartime_energy_window': neartime_energy_window,
                    'market_interval': market_interval,
                     "market_window_length": market_window_length,
                     "market_resolution": market_resolution})
    return market_design_parameters




def getFairnessFactorWeights():
    historic_success_weight=0.6
    unused_weight=0.4

    fairness_factor_weights = dict({'historic_success': historic_success_weight,
                                   'unused_weight': unused_weight})
    
    return fairness_factor_weights

def getAvailableProductsList():
    products_list = ['premium', 'basic']
    return products_list

def getProductInfo(product_type):
    
    if product_type == 'premium':
        product_cost = getPremiumProductInfo().get("premium_product_cost")
        product_success = getPremiumProductInfo().get("premium_product_success")
    elif product_type == 'basic':
        product_cost = getBasicProductInfo().get("basic_product_cost")
        product_success = getBasicProductInfo().get("basic_product_success")

    product_parameters = dict({'product_cost': product_cost,  
                               'product_success': product_success})

    return product_parameters


def getBasicProductInfo():
    basic_product_cost = 1 #cost per day
    basic_product_weight = 1

    basic_product_info = dict({'basic_product_cost': basic_product_cost,  
                            'basic_product_weight': basic_product_weight

    })
    return basic_product_info

def getPremiumProductInfo():
    premium_product_cost = 2 #cost per day
    premium_product_weight = 2

    premium_product_info = dict({'premium_product_cost': premium_product_cost,  
                            'premium_product_weight': premium_product_weight

    })
    return premium_product_info


def getDeadbandPercentage():
    deadband_percentage = 0.3
    return deadband_percentage




