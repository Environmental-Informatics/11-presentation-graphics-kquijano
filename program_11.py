#!/bin/env python
# Add your own header comments
#

"""
Karoll Quijano - kquijano

ABE 651: Environmental Informatics

Assignment 11
Presentation Graphics

This script servesa as the solution set for assignment-11 on presentation graphics using
descriptive statistics and environmental informatics. 

"""


import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt



def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    #  remove negative streamflow values as a gross error check
    DataDF.loc[(DataDF['Discharge']<0)]=np.nan 
            
    # Number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )


def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""

    # Clip data 
    DataDF = DataDF[startDate:endDate]
    # Count missing values 
    MissingValues = DataDF["Discharge"].isna().sum()    
        
    return( DataDF, MissingValues )


def ReadMetrics( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    the metrics from the assignment on descriptive statistics and 
    environmental metrics.  Works for both annual and monthly metrics. 
    Date column should be used as the index for the new dataframe.  Function 
    returns the completed DataFrame."""
    
    # Read file
    DataDF = pd.read_csv(fileName, header=0, delimiter=',',
                         parse_dates=['Date'], index_col=['Date']) 
                        
    return( DataDF )



# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define full river names as a dictionary so that abbreviations are not used in figures
    riverName = { "Wildcat": "Wildcat Creek",
                  "Tippe": "Tippecanoe River" }
    
 
       
    # File names 
    files = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    metrics = {"Annual": "Annual_Metrics.csv", 
               "Monthly": "Monthly_Metrics.csv"}
    
    
    # New dictionaries 
    DataDF = {} 
    MissingValues = {}
    MetricsDF = {}
    
    
    # runthrough txt files to get data
    for file in files.keys():
        DataDF[file], MissingValues[file] = ReadData(files[file])
        DataDF[file], MissingValues[file] = ClipData(DataDF[file], '2014-10-01' , '2019-09-30')
        #MoDataDF[file] = GetMonthlyStatistics(DataDF[file])
        #MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
    
    #run through csv to get metrics
    for metric in metrics.keys(): 
        MetricsDF[metric] = ReadMetrics(metrics[metric])

    
# Daily flow for both streams for the last 5 years of the record.
    plt.plot(DataDF['Wildcat']['Discharge'], color='b', label='Wildcat Creek', linewidth=1)
    plt.plot(DataDF['Tippe']['Discharge'], color='r', label='Tippecanoe River', linewidth=1)
    plt.legend(loc='upper right')
    plt.title('Daily Flow for Last 5 Years')
    plt.xlabel('Year')
    plt.ylabel('Discharge (cfs)')
    plt.savefig('DailyFlow.png', dpi = 300)
    plt.close()    

# Annual coefficient of variation, Tqmean, and R-B index.
    fig = plt.figure(figsize = (10,7)) 
    
    plt.subplot(311)
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Wildcat']['Coeff Var'],  color='b', label='Wildcat Creek')
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Tippe']['Coeff Var'], color='r', label='Tippecanoe River')   
    plt.ylabel('Coefficient of Variation')
    plt.title('Annual Coefficient of Variation, Tqmean, and R-B index')
    plt.legend(loc='upper right')
    
    # Tqmean
    plt.subplot(312)
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Wildcat']['Tqmean'],  color='b')
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Tippe']['Tqmean'], color='r') 
    plt.ylabel('Tqmean')
    
    # R-B index.
    plt.subplot(313)
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Wildcat']['R-B Index'],color='b')
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Tippe']['R-B Index'],color='r')  
    plt.ylabel('R-B Index')
    
    plt.legend()
    plt.xlabel('Date')
    plt.subplots_adjust(hspace= 0.4)
    plt.savefig('AnnualCoefficients.png',dpi=300)
    plt.close()
 

# Average annual monthly flow (so 12 monthly values, maybe you need an additional function from program-10.py?).
    MoMetricsDF = MetricsDF['Monthly'].groupby('Station')
    color = {"Wildcat":'b', "Tippe":'r'}
    for name, data in MoMetricsDF:
        cols=['Mean Flow']
        m=[3,4,5,6,7,8,9,10,11,0,1,2]
        index=0
        # New dataframe
        MonthlyAverages=pd.DataFrame(0,index=range(1,13),columns=cols)
        # New output table
        for i in range(12):
            MonthlyAverages.iloc[index,0]=data['Mean Flow'][m[index]::12].mean()
            index+=1
        plt.plot(MonthlyAverages.index.values, MonthlyAverages['Mean Flow'].values, label=riverName[name],color=color[name])    
    plt.legend()
    plt.title('Average Annual Monthly Flow')
    plt.ylabel('Discharge (cfs)')
    plt.xlabel('Month')
    plt.savefig('AverageAnnualMonthlyFlow.png', dpi=300)
    plt.close()
    
    
# Return period of annual peak flow events
    PeakFlow = {}
    
    for file in files.keys():
        PeakFlow[file] = MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']==file]['Peak Flow'].sort_values(ascending=False)
        E_P =[]
        N = len(PeakFlow[file])
        for i in range(1,N+1):
            E_P.append(i/(N+1))
        
        PeakFlow[file].index = pd.Series(E_P)
        PeakFlow[file].index.name = 'Exceedence Probability'

    # create the plot of return period of annual peak flow events for both streams
    
    PeakFlow['Wildcat'].plot(color='b',linestyle='None',marker='.')
    PeakFlow['Tippe'].plot(color='r', linestyle='None',marker='.')
    plt.legend([riverName['Wildcat'],riverName['Tippe']])
    plt.title("Return Period of Annual Peak Flow Events")
    plt.xlabel("Exceedence Probability")
    plt.ylabel("Discharge (cfs)")
    plt.savefig("ReturnPeriodAnnualPeak.png",dpi=300)
    plt.close()
