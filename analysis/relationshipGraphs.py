# Author: Shaun Sweeney 
# Contact: shaunsweeney12@gmail.com
# Initial creation: March 2024

# Purpose: 

# This script produces some figures and gathers results using the raw market outcomes. 


import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import graphFunctions as gF
import time
import pricingFunctions
import plotly.express as px
import seaborn as sns


def translateRequestsToSparseTsTest(rows):
    periods=[]
    rows.reset_index(inplace=True)

    for index, row in rows.iterrows():

        start_time = row['Accepted Start Time']
        end_time = row['Accepted End Time']
        date_index = pd.date_range(start=start_time,end=end_time,freq="5Min")

        period_ts = pd.DataFrame(np.zeros(len(date_index)), index=date_index, columns=['Grid Health'])
        period_ts['Power'] = row['Request Power']
        period_ts['Grid Health'] = row['Market Average Grid Health'] 
  
        periods.append(period_ts)

    result_ts = pd.concat(periods)

    return result_ts


def createBoxPlots(scenarios_list,keywords):
    boxplot_df=pd.DataFrame()
    column_names = []

    for scenario_key, scenario_values in scenarios_list.items():
        boxplot_df['{}'.format(scenario_key)] = scenario_values
        column_name = '{}'.format(scenario_key)
        column_names.append(column_name)


    box_plot = boxplot_df.boxplot(column=column_names, showfliers=False, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))  

    plt.style.use("bmh")
    plt.ylabel('Average unit cost (£/kWh)')
    plt.tight_layout()
    plt.savefig('figures/'+keywords[0]+'_box'+str(time.time())+'.png')
    plt.show()


def createBarGraph(energy_allocations_dict):
    approach_1_total = energy_allocations_dict.get("Approach 1_total_percentage_served")*100
    approach_1_high_willingness_to_pay = energy_allocations_dict.get("Approach 1_high_willingness_to_pay_percentage")*100
    approach_1_low_init_success = energy_allocations_dict.get("Approach 1_low_init_fairness_percentage")*100

    approach_2_total = energy_allocations_dict.get("Approach 2_total_percentage_served")*100
    approach_2_high_willingness_to_pay = energy_allocations_dict.get("Approach 2_high_willingness_to_pay_percentage")*100
    approach_2_low_init_success = energy_allocations_dict.get("Approach 2_low_init_fairness_percentage")*100

    approach_3_total = energy_allocations_dict.get("Approach 3_total_percentage_served")*100
    approach_3_high_willingness_to_pay = energy_allocations_dict.get("Approach 3_high_willingness_to_pay_percentage")*100
    approach_3_low_init_success = energy_allocations_dict.get("Approach 3_low_init_fairness_percentage")*100

    species = ("Approach 1", "Approach 2", "Approach 3")
    penguin_means = {
        'Total': (approach_1_total, approach_2_total, approach_3_total),
        'High willingness to pay': (approach_1_high_willingness_to_pay, approach_2_high_willingness_to_pay, approach_3_high_willingness_to_pay),
        'Low initial success': (approach_1_low_init_success, approach_2_low_init_success, approach_3_low_init_success),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage of energy requested that is delivered')
    ax.set_title('Allocation of energy in shortage event')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=1)
    ax.set_ylim(0, 105)

    plt.savefig('figures/bar'+str(time.time())+'.pdf')

    plt.show()




keywords = ["fairaccess"]
# keywords = ["behaviours"]

use_biased_init_fairness = False
use_bised_init_buy_prices = True
process_by_neighbour=False

if ("fairaccess" in keywords):
    rootfolder='results/'


    #0 flex offered - 17/12/2021 - 18/12/2021 - 2500000 - biased init buy prices and fairness
    results_volume = pd.read_csv(rootfolder+'debuggingvolume1717366086_scale250000flex_0.csv', index_col=0)
    results_price = pd.read_csv(rootfolder+'debuggingprice1717365957_scale250000flex_0.csv', index_col=0)
    results_queue= pd.read_csv(rootfolder+'debuggingqueue1717366054_scale250000flex_0.csv', index_col=0)

    #lookup
    # Approach 1: volume_maximising
    # Approach 2: revenue_maximising
    # Approach 3: with_fairness


    raw_results_list = dict({"Approach 1": results_volume,
            "Approach 2": results_price,
            "Approach 3": results_queue})
    


elif("behaviours" in keywords):
    rootfolder='finalresults/behaviourrewarded/'

    #125000 - october only - 31 day market window length
    results_0_flex = pd.read_csv(rootfolder+'debuggingqueue1713120217_scale125000flex_0.csv', index_col=0)
    results_3_flex = pd.read_csv(rootfolder+'debuggingqueue1713120798_scale125000flex_3.csv', index_col=0)
    results_6_flex = pd.read_csv(rootfolder+'debuggingqueue1713121398_scale125000flex_6.csv', index_col=0)
    results_12_flex = pd.read_csv(rootfolder+'debuggingqueue1713122016_scale125000flex_12.csv', index_col=0)


    raw_results_list = dict({"0 hours": results_0_flex,
                             "3 hours": results_3_flex,
                            "6 hours": results_6_flex,
                            "12 hours": results_12_flex})




grid_health_series=[]
neigbours_vals_to_plot=[]
neighbour_power_ts = []
series_to_plot_list=[]


series_to_plot = dict()
per_unit_costs_dict = dict()
low_init_fairness_list = [] 

# read this in for biased prices
neighbourFolder='files/BuyPrices/'
neighbour_filename = neighbourFolder+'neighbours1713127303.pickle'
neighbours_list = pd.read_pickle(neighbour_filename)


if("behaviours" in keywords):

    neigbours_vals_to_plot=[]
    per_unit_costs = []

    unique_neighbour_vals = results_0_flex['Neighbour ID'].unique()

    for results_key, results in raw_results_list.items():

        new_data={}
        per_unit_costs=[]

        for neighbour_number in unique_neighbour_vals:

            current_neighbour_rows = results[(results['Neighbour ID']==neighbour_number) & (results['Request Acceptance']==True)]
 
            current_neighbour_cost = current_neighbour_rows['Request Cost'].sum()
            current_neighbour_energy = current_neighbour_rows['Request Total Energy Delivered'].sum()

            current_neighbour_per_unit_cost = (current_neighbour_cost/current_neighbour_energy)/100

            current_neighbour_vals = {'neighbour_id': neighbour_number,
                                    'current_neighbour_cost':current_neighbour_cost, 
                                    'current_neighbour_energy': current_neighbour_energy,
                                    'current_neighbour_per_unit_cost': current_neighbour_per_unit_cost}
            
            neigbours_vals_to_plot.append(current_neighbour_vals)

            per_unit_costs.append(current_neighbour_per_unit_cost)

        df_to_plot = pd.DataFrame(neigbours_vals_to_plot)
        df_to_plot.set_index("neighbour_id", inplace=True)
        df_to_plot.sort_index(ascending=True, inplace=True) 

        per_unit_costs.sort()

        new_data = {results_key: per_unit_costs}
        per_unit_costs_dict.update(new_data)


    createBoxPlots(per_unit_costs_dict,keywords)



elif("fairaccess" in keywords):

    per_unit_costs = []
    unique_neighbour_vals = results_volume['Neighbour ID'].unique()
    neighbour_fairness_dict={}
    energy_allocations_dict={}


    for results_key, results in raw_results_list.items():
        neigbours_vals_to_plot=[]
        series_to_plot={}
        new_data={}
        neighbour_fairness=[]


        total_energy_desired=0
        total_energy_delivered=0
        high_willingness_to_pay_energy_desired=0
        high_willingness_to_pay_energy_delivered=0
        low_init_fairness_value_energy_desired=0
        low_init_fairness_value_energy_delivered=0


        for ind in results.index:
            total_energy_desired += results['Request Total Energy Desired'][ind]
            total_energy_delivered += results['Request Total Energy Delivered'][ind]

            #convention used for experiment was that all even numbered neighbour IDs had a low initial history of request success and all odd numbered neighbours had a high willingness to pay
            neighbour_id = results['Neighbour ID'][ind]
            if((neighbour_id % 2)==0):
                is_even=True
            else:
                is_even=False

            if(is_even==False):
                high_willingness_to_pay_energy_desired += results['Request Total Energy Desired'][ind]
                high_willingness_to_pay_energy_delivered += results['Request Total Energy Delivered'][ind]

            elif(is_even == True):
                low_init_fairness_value_energy_desired += results['Request Total Energy Desired'][ind]
                low_init_fairness_value_energy_delivered += results['Request Total Energy Delivered'][ind]


        total_percentage_served = round(total_energy_delivered/total_energy_desired, 2)
        high_willingness_to_pay_percentage = round(high_willingness_to_pay_energy_delivered/high_willingness_to_pay_energy_desired,2)
        low_init_fairness_percentage = round(low_init_fairness_value_energy_delivered/low_init_fairness_value_energy_desired,2)

        new_data = {results_key+'_total_percentage_served': total_percentage_served,
                        results_key+'_high_willingness_to_pay_percentage':high_willingness_to_pay_percentage, 
                        results_key+'_low_init_fairness_percentage': low_init_fairness_percentage}
            

        energy_allocations_dict.update(new_data)
            

    createBarGraph(energy_allocations_dict)

