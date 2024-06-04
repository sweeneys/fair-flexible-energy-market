# REPO DESCRIPTION / OBJECTIVE 

- This repo is for allocating energy according to the methodology described in [link to paper].

- It is required to provide input datasets for supply and consumption. Consumption should be split into essential consumption and flexible consumption which can be done using repo https://github.com/sweeneys/consumption-characteriser.


# SCRIPTS 

- configInputs.py: Executable used to set configuration options for the energy market.

- energyMarket.py: Executed by configInputs.py.

- marketDesignFunctions.py: Contains functions specifying parameters relating to the market design.

- marketFunctions.py: Contains functions required for the execution of the market.

- pricingFunctions.py: Contains functions relating to the pricing function which determine the buy and sell price for energy.

- standardiseInputs.py: Is used to relate the supply dataset to the consumption dataset based on a number of different methodologies.

- analysis/relationshipGraphs.py: This is a script for producing some figures and headline stats after the market results are available.


# INSTRUCTIONS 

1. Add the files containing the neighbourParameters.pickle, requests.pickle, the file with a prediction of the final consumption (this can be generated using the requests depending on the forecasting methodology in use), a file with the supply input, and a file with the essential consumption (if using). 

NOTE: If consumption files are not available, these can be generated using https://github.com/sweeneys/consumption-characteriser. Further info is given in the README.md for this repo. 

2. Set config parameters

 - start_time and end_time: This is the global start and end time for the market to execute. The consumption and supply datasets will be spliced to find the relevant periods. 

 - execute_multiple_flex_levels: Set as True or False, if true this provides an opportunity to input a number of different request files which will repeat the market execution for several different flexibility levels.
 
 - appliances_labelled: Set as True or False, this relates to whether appliances are labelled in the input consumption dataset or not. This determines whether the fairness policy is updated at the appliance level (if labelled) or the household level (if not). 
  
 - essential_consumption_type: Set to none if it is desired to assume that the whole supply dataset input is available for use by flexible appliances, or if it desired to subtract essential consumption from supply first, set to "timeseries". Note: Ensure that a link to the essential consumption dataset is provided. 
   
 - resource_allocation_method: Set to "queue", "volumeoptimisation" or "priceoptimisation". These correspond with different ways that the resource allocation methodology is done.  "queue" is the novel stochastic dynamic resource allocation methodology which allocates resource in line with a specified fairness policy. "volumeoptimisation" maximises the global use of the resource and is revenue agnostic, and "priceoptimisation" is a global optimisation which maximises the revenue earned. 
 
 - repeat_same_day_multiple_times and number_of_daily_repeats: This is useful for experimentation with the queuing approach which is stochastic and to see how outcomes evolve over time while removing supply variability over time
  
 - scaling_methodology - method: Choose method to relate the consumption and supply datasets, further information on the different methodologies is given in the standardiseInputs.py file. 
 
 - scaling_methodology - value: If chosen "scale_to_fixed_value" as the method above, the value inputted here is the value used to scale (down) the supply. 
 
 - plot_live_graph: Set to True if it is desired to plot a video of the real time processing of the market for the queuing approach
 
 

3. Execute configInputs.py in a Python Terminal


# OUTPUTS 

1. CSV file outputting the status of requests after execution of filename: debugging{allocationmechanism}{timestamp}_scale{supplyconsumptionratio}flex_{flexibilitylevel}.csv
NOTE: This is raw output data which may be interesting to look at, the analysis/relationshipGraphs.py script provides some tools useful to interpret this.

2. CSV file on supply remaining available at end of execution: Useful for debugging how much supply was used during the market instance. 

3. Code profiling .txt file: Stats on performance of code



# LICENSE
IP is owned by Shaun Sweeney and Imperial College London. This codebase is open-access can be freely used for experimentation or for other projects to be built on it. All work developed based on this should correctly acknowledge and cite the original author. 


# CONTACT
Contact shaunsweeney12@gmail.com


# ACKNOWLEDGMENTS
This work was developed as part of PhD research at Dyson School of Design Engineering, Imperial College London. 
