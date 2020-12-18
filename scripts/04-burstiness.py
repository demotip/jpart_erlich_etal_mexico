
"""
    Performs burtiness detection using Kleinberg's algorithm
"""

import pandas as pd
import numpy as np
import settings
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import time
from datetime import date
import calendar
from algos.burst_detection import burst_detection,enumerate_bursts

#======================= Change Parameters here =======================#
NEWSPAPER = settings.SOURCE_NEWSPAPER
PREPROCESSED_DF = '../data_new/df_comprehensive_REGEX.pkl'
#======================================================================#
df_agency_coverage = pd.read_pickle(PREPROCESSED_DF)[pd.read_pickle(PREPROCESSED_DF)['source'] == NEWSPAPER]



df_agency_coverage_w = df_agency_coverage.set_index('date').resample('w',label='left').sum().iloc[:,1:-1]
df_agency_coverage_d = df_agency_coverage.set_index('date').resample('d',label='left').sum().iloc[:,1:-1]

df_agency_coverage_mon = df_agency_coverage.set_index('date').resample('w-mon',label='left').sum().iloc[:,1:-1]
df_agency_coverage_tue = df_agency_coverage.set_index('date').resample('w-tue',label='left').sum().iloc[:,1:-1]
df_agency_coverage_wed = df_agency_coverage.set_index('date').resample('w-wed',label='left').sum().iloc[:,1:-1]
df_agency_coverage_thu = df_agency_coverage.set_index('date').resample('w-thu',label='left').sum().iloc[:,1:-1]
df_agency_coverage_fri = df_agency_coverage.set_index('date').resample('w-fri',label='left').sum().iloc[:,1:-1]
df_agency_coverage_sat = df_agency_coverage.set_index('date').resample('w-sat',label='left').sum().iloc[:,1:-1]


agency_names = df_agency_coverage_w.columns.values

#---- Collect dataframes together ----#
collection = [(df_agency_coverage_sat, "Saturday"),
                      (df_agency_coverage_w,   "Sunday"),
                      (df_agency_coverage_mon, "Monday"), 
                      (df_agency_coverage_tue, "Tuesday"),
                      (df_agency_coverage_wed, "Wednesday"),
                      (df_agency_coverage_thu, "Thursday"),
                      (df_agency_coverage_fri, "Friday")
                     ]

datelist = pd.date_range('2004-12-25', '2016-12-30').tolist()  # contains every day in dataframe
datelist = [time.to_datetime64() for time in datelist]         # each date is of type numpy.datetime64



#---- Some further processing ----#
df_start_times = []
for agency_name in df_agency_coverage_w.columns.values: # iterate over all agencies
    
    agency_data = []     
    for start_time_df in collection: # iterate over the collection of data frame that vary be week start time
        df = start_time_df[0]
        agency_mentions = df[agency_name].values
        agency_data.append(agency_mentions)

    agency_data = np.array(agency_data)
    agency_data = np.transpose(agency_data)
    # we'll add the agency name to each column name to make it easier to remember which agency's data you're viewing
    
    headers = []
    days = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for day in days:
        day_w_agency = agency_name + ": " + day
        headers.append(day_w_agency)
    
    agency_df = pd.DataFrame(data = agency_data, columns=headers)
    
    agency_df.name = agency_name
    
    df_start_times.append(agency_df)



RESIDUAL = False

#TODO: implement residuals

#---- Seasonal Decomposition ----#

'''
    Applies a seasonal decompose to a df.
    2. Seasonal decompose
    
    Args:
        df : pd.DataFrame
        
    Returns:
        tuple:
            df_seasonal : pd.DataFrame
            df_trend : pd.DataFrame
            df_residual : pd.DataFrame
'''

"""
if RESIDUAL:
    def seasonal_decompose_df(df_tup):
        df = df_tup[0]
        df_residual = df.copy()

        for col in df.columns:
            d = seasonal_decompose(df[col],two_sided=False)
            df_residual[col] = d.resid
        return (df_residual)

    #---- Apply seasonal decomposition ----#
    collection_residual = []
    for df in collection:
        (df_processed_residual) = seasonal_decompose_df(df)
        collection_residual.append(df_processed_residual - df_processed_residual.min())
        
    collection = collection_residual

"""
# for the burst function
#inputs:
#   r: number of target events in each time period (1xn)
#   d: number of events in each time period (1xn)
#   n: number of timepoints
#   s: multiplicative distance between states
#   gamma: difficulty to move up a state
#   smooth_win: width of smoothing window (use odd numbers)
#output:
#   q: optimal state sequence (1xn)
#   d: number of events in each time period
#   r: number of target events in each time period
#   p: expected proportion for each state
    


n = len(df_agency_coverage_w)
output_burst = []
s = 1.5 #
gamma = 1.0 #
for day_index, start_time_df in enumerate(collection):
    df = start_time_df[0]
    output = {}
    for col_index, agency in enumerate(df.columns):
        [q, _, _, p] = burst_detection(df[agency],df.sum(axis=1),n,s,gamma,smooth_win=3)
        
        burst_out = enumerate_bursts(q, '')
        out = burst_out[['begin','end']].values.flatten().tolist()
        if len(out) > 0:
            output[agency] = out
        

    output_burst.append((days[day_index], output))


ESD_output_by_start_day = output_burst



#ESD_output_collection has a dict for each of the seven start days, each dict contains the anomalies detected for each agency

ESD_output_by_agency = []
for agency in agency_names:
    agency_data = [agency]
    anomaly_dict = {}
    for index, ESD in enumerate(ESD_output_by_start_day):
        if agency in ESD[1].keys():
            anomaly_dict[days[index]] = list(sorted(ESD[1][agency]))
    agency_data.append(anomaly_dict)
    ESD_output_by_agency.append(agency_data)
    

# _anomaly_comparison_collection lets you compare an agency's mentions by start day    
_anomaly_comparison_collection = []
for name in agency_names:
    agency_data = []
    for df in collection:
        df = df[0]
        dates = pd.to_datetime(df.index.values)
        dates = [date.date() for date in dates]
        
        agency_data.append(dates)
        agency_data.append(list(df[name].values))
        
    agency_data = np.array(agency_data)
    new_df = pd.DataFrame(data=agency_data.transpose(), 
                          columns = ['1', 'Saturday', '2', 'Sunday','3', 'Monday','4', 'Tuesday','5', 'Wednesday','6', 'Thursday','7', 'Friday'])
    _anomaly_comparison_collection.append(new_df)
    
#
def color_anomalies_by_agency(agency, anomaly_dict = ESD_output_by_agency[0][1]): 
    col_names = ['1', '2', '3', '4', '5', '6', '7']
    if agency.name not in col_names:         
        anomaly_list = anomaly_dict[agency.name]
        colormap = ['background-color: #ff3333' if index in anomaly_list else '' for index, row in enumerate(agency)]

    else:
        colormap = ['background-color: #d9d9d9', 'background-color: #f2f2f2'] * int(len(agency)/2)
        colormap.append('background-color: #d9d9d9')

    return colormap

anomaly_comparison_colored_by_agency = []
for index, df in enumerate(_anomaly_comparison_collection):
    anomaly_comparison_colored_by_agency.append(df.style.apply(color_anomalies_by_agency, anomaly_dict=ESD_output_by_agency[index][1]))

    
    
    
#----- Combine all of the data from each start day and find the exact date ranges of the anomalies ----#
datelist = pd.date_range('2004-12-25', '2016-12-30').tolist()  # contains every day in dataframe
datelist = [time.to_datetime64() for time in datelist]         # each date is of type numpy.datetime64

def sortTup(tuplist):
     return sorted(tuplist, key=lambda x: x[0])
    
def mergeIntervals(interval_list):
    anomaly_ranges_final = []
    if len(interval_list) > 1:
        front = interval_list[0][0]
        back  = interval_list[0][1]

        for interval in interval_list:
            if interval[0] >= front and interval[0] <= back: # start of interval in previous interval range
                if interval[1] > back: #end of this interval greater than current back
                    back = interval[1]
                else: # entire interval contained in current range, do nothing
                    pass

            else: # beginning of cur interval greater than range of front and back, we have a new interval
                anomaly_ranges_final.append((front, back)) # we add old range to our interval list and begin a new interval
                front = interval[0]
                back = interval[1]

        anomaly_ranges_final.append((front, back))
        return anomaly_ranges_final


anomalies_foreach_agency = []

for agency_data in ESD_output_by_agency:    
    anomalies = agency_data[1]
    all_agency_intervals = [agency_data[0]]
    interval_list = []

    for week_index, week in enumerate(anomalies):        
        anomaly_list = anomalies[week]
        
        for anomaly in anomaly_list: 
            # we're going to get a list of indices from the datelist for each anomaly
            # e.g. Saturday: index 250 will map to datelist[1750]
            tup = collection[week_index]
            df = tup[0]                         # dataframe corresponding to the week_index
            date = df.index.values[anomaly]
            datelist_index = datelist.index(date)

            indices = (datelist_index, datelist_index+6) # indices of the datelist that encompany the anomaly week

            interval_list.append(indices)
            

    all_agency_intervals.append(mergeIntervals(sortTup(list(set(interval_list))))) # appends list of indices, sorted by 1st element, merged, with duplicates removed    
    anomalies_foreach_agency.append(all_agency_intervals)

    # now we want to smooth the intervals to a find a date range that encompasses the whole anomaly


anomaly_dates_foreach_agency = []
for anomalies in anomalies_foreach_agency:
    agency_name = anomalies[0]
    anomalies = anomalies[1]
    if anomalies:
        anomaly_dates = [agency_name]
        for anomaly in anomalies:
            date_start = pd.to_datetime(datelist[anomaly[0]]).date()
            date_end = pd.to_datetime(datelist[anomaly[1]]).date()
            date_range = (date_start, date_end)
            anomaly_dates.append(date_range)
        anomaly_dates_foreach_agency.append(anomaly_dates)
    
    
#---- Building dataframe containing anomalies ----#

start_dates = []
end_dates = []
name_list = []
for agency in anomaly_dates_foreach_agency:
    agency_name = agency[0]
    agency_dates = agency[1:]    
    
    
    for date in agency_dates:
        start_dates.append(date[0])
        end_dates.append(date[1])
        name_list.append(agency_name)
        
anomaly_df = pd.DataFrame(data = [name_list,start_dates, end_dates])
anomaly_df = anomaly_df.transpose()
anomaly_df.columns = ['Agency', 'Start Date', 'End Date']
anomaly_df = anomaly_df.sort_values(by=['Start Date'])
anomaly_df = anomaly_df.reset_index(drop=True)

anomaly_df.to_pickle(settings.DATA_NEW+"burstiness_date_df_"+str(s)+"_"+str(gamma)+"_.pkl")





