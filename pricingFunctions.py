
# Author: Shaun Sweeney 
# Contact: shaunsweeney12@gmail.com
# Initial creation: December 2022

# Purpose: 

# This script is designed to enable users to specify parameters relating to pricing functions i.e. how much someone pays for their flexible consumption. 
# Currently this  contains a linear pricing function but this can easily be extended to other desired pricing functions. 

def linearPricingFunction():
    buy_price_slope=-2.65
    sell_price_slope=-2.65
    buy_price_y_intercept=265
    sell_price_y_intercept=0 

    pricing_function = dict({'buy_price_slope': buy_price_slope,
                        'sell_price_slope': sell_price_slope,
                        'buy_price_y_intercept':buy_price_y_intercept,
                        'sell_price_y_intercept':sell_price_y_intercept})
    
    # this is to make a graph of buy and sell price curves
    # min_grid_health_value=0
    # max_grid_health_value=100
    # grid_health_values = range(min_grid_health_value, max_grid_health_value+1)
    # energyBuyPriceCurve = np.dot(buy_price_slope, grid_health_values) + buy_price_y_intercept
    # energySellPriceCurve = np.dot(sell_price_slope, grid_health_values) + sell_price_y_intercept

    # pricingFunction=[]
    # pricingFunction.append(grid_health_values, energyBuyPriceCurve, energySellPriceCurve)


    return pricing_function