import yfinance as yf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats as sps
import matplotlib.pyplot as plt



#main filepath
filepath = 'D:\\Python_DOCS\\STOCK_CSVs\\'

#extract symbol from ETF holdings file
#symPATH = 'D:\\Stock_stuff\\OO_DIVS.csv'
#symbols = pd.read_csv(symPATH)
#symbols.info()
symbols = ["list of stock tickers goes here"]# or you can import from a list from csv etc...


##########################
#### IMPORT DATA FUNCTION
def IMPORT_DATA(ticker, start0, end0 , intvl):

        if not ticker or not start0 or not end0 or not intvl: print("Nothin'")
        elif ticker:
            #Import Yahoo Data and store in CSV then edit text to collumns
            DF = yf.Ticker(ticker)
            print(ticker)
            DF_data = pd.DataFrame(DF.history(start = start0, end = end0, interval = intvl))
            ## place insert 'Index' code here...
            #print(DF_data)
            DF_data.info()
            print("df type is -> ",type(DF_data))
            DF_data.to_csv(ticker+'.csv', index=True)
            return pd.read_csv(ticker+'.csv')


start0 =  "2026-01-22"
end0 = "2026-01-23"
intvl = "1m"

SYMBOL = []
probPOS = []
probNEG = []
Mean = []
StandardDeviation = []
Variance = []
Skewness = []
Kurtosis = []
DivYIELD = []
Name = []
Sector = []


for s in symbols:#['Symbol']:

    df = IMPORT_DATA(s, start0, end0 , intvl)
    dates = df.index
    print("")
    
    X = np.array(df['Close'].dropna())#closing prices
    #print(X)
    
    PercChng = [0]
    portionPOS = 0
    portionNEG = 0
    
    
    for i in range(0,len(X)-1): 
        PercChng.append((X[i+1]-X[i])/X[i])
        if PercChng[i-1] >0: portionPOS= portionPOS+1
        elif PercChng[i-1] <=0: portionNEG = portionNEG+1
    
    PercChng = np.array(PercChng)#conver to array
    
    portionPOS = portionPOS/len(X)
    portionNEG = portionNEG/len(X)
    sDev = np.std(PercChng)#standard deviation PC's
    var = np.var(PercChng)#variance PC's


    SYMBOL.append(s)
    probPOS.append(portionPOS)
    probNEG.append(portionNEG)
    Mean.append(np.mean(PercChng))
    StandardDeviation.append(np.std(PercChng))
    Variance.append(np.var(PercChng))
    Skewness.append(sps.skew(PercChng)) 
    Kurtosis.append(sps.kurtosis(PercChng))
    #DivYIELD.append(str(float(100*(symbols['Dividend Yield'][symbols['Symbol']==s])))   )
    #Name.append(str((symbols['Name'][symbols['Symbol']==s].iloc[0])))
    #Sector.append(str((symbols['Sector'][symbols['Symbol']==s].iloc[0])))

    

    
    
pc_data = {'Symbol':SYMBOL,'probPOS': probPOS,'ProbNEG':probNEG,'Mean':Mean,
           'StandardDeviation':StandardDeviation,'Variance':Variance,'Skewness':Skewness,
           'Kurtosis':Kurtosis   } #'Sector':Sector,

# pc_data = {'Symbol':SYMBOL,'probPOS': probPOS,'ProbNEG':probNEG,'Mean':Mean,
#            'StandardDeviation':StandardDeviation,'Variance':Variance,'Skewness':Skewness,
#            'Kurtosis':Kurtosis  } #'Sector':Sector,


pc_DF = pd.DataFrame(pc_data).dropna()# removes rows with null data
pc_DF.info()
print(pc_DF)



holdpath = 'D:\\Stock_stuff\\'
pc_DF.to_csv('Multi_STOCK_STATS2.csv', index=True)

POS_mean =  np.mean(pc_DF['probPOS'])

print('average positive return probability = ',POS_mean)

HI_POTENTIAL_DF = pc_DF
    
        
HI_POTENTIAL_DF = HI_POTENTIAL_DF[HI_POTENTIAL_DF['probPOS']>0.55]

HI_POTENTIAL_DF.to_csv('HI_POTENTIAL.csv', index=True)

#print(pd.unique(HI_POTENTIAL_DF['Sector']))



