import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft
from scipy.signal import blackman
from sklearn.preprocessing import LabelEncoder
import statsmodels.tsa.stattools as stt


def make_data(freq,periods, ncol=1):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    idx = pd.date_range(now,freq=freq,periods=periods)
    x = np.linspace(0, 100, num=periods)
    pi2 = 2.*np.pi    
    valDic = {}
    for i in range(ncol):
        value = np.random.randint(10)*np.sin(0.1*pi2*x) +np.random.randint(10)*np.cos(1.*pi2*x) +  np.random.rand(1)*np.random.randn(x.size)
        inx = "val" + str(i)
        valDic[inx] = value    
    df = pd.DataFrame(valDic,index=idx)
    return df

def make_anomaly(df, periods, num, direction, trg_col=[]):
    start = np.random.randint(periods)
    end = start + num
    df_anom = df

    if trg_col == []:trg_col = df.columns
    for i in trg_col:
        if direction == "pos":
            df_anom[i].iloc[start:end] = float(df_anom[i].max()) * 10
        elif direction == "neg":
            if float(df_anom[i].min()) < 0:
                df_anom[i].iloc[start:end] = float(df_anom[i].min()) * 10
            else:
                df_anom[i].iloc[start:end] = float(df_anom[i].min()) / 10
        df_anom = df_anom.rename(columns={i: i+"_anom"})
    return df_anom

def resample_time(df, freq, how="mean"):
    if how == "last":
        df = df.resample(freq).last()
    elif how == "first":
        df = df.resample(freq).first()
    elif how == "sum":
        df = df.resample(freq).sum()
    elif how == "median":
        df = df.resample(freq).median()
    elif how == "max":
        df = df.resample(freq).max()
    elif how == "min":
        df = df.resample(freq).min()        
    else:#mean
        df = df.resample(freq).mean()

    return df

def missing_value(df, method="interpolate", _na_ratio=0.2):
    if method == "interpolate": #fill na by using linear interpolate
        df = df.interpolate()
    elif method == "drop": #drop na
        original_length = len(df)
        cols = df.columns
        for col in cols:
            tmp_df = df.dropna(subset=[col], how="any")
            tmp_length = len(tmp_df)
            na_ratio = float(original_length - tmp_length) / original_length
            if na_ratio <= _na_ratio:
                df = tmp_df
                original_length = len(df)
            else:
                df = df.drop(col, axis=1)
    else:
        df = df.fillna(df.mean()) #fill na with mean value
    return df

#convert timestamp format into yyyy/mm/ddTHH:mm:ss
def index_convert(df):
    index = df.index
    converted_index = []
    for i in index:
        j = "T".join(str(i).split(" ")) + "+09:00"
        converted_index.append(j)
    df.index = converted_index
    return df

def calc_return(df):
    rets = []
    for i in  range(len(df.columns)):
        ret = df.iloc[:,i].pct_change().dropna()
        rets.append(ret)
    
    return rets

def calc_pacf(df):
    acfs = []
    pcfs = []
    for i in range(len(df.columns)):
        acf = stt.acf(df.iloc[:,i])
        acfs.append(acf)
        
        pcf = stt.pacf(df.iloc[:,i])
        pcfs.append(pcf)
    
    return acfs, pcfs

# get moving average
def calc_ma(df):
    mas = []
    for i in range(len(df.columns)):
        ma = pd.Series.rolling(df.iloc[:,i], window=3, center=True).mean()
        mas.append(ma)
    return mas

def calc_fft(df):
    # Number of sample points
    N = len(df)
    # sample spacing
    T = 1.0 / N * 1.5
   ##window function
    w = blackman(N)
    y = df
    yf = 2.0/N * np.abs(fft(y*w)[0:N//2])
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    return yf, xf

def label_enc(df, label):
    le = LabelEncoder()
    le.fit(df[label])
#     print("#####" + label + "#####")
#     print(le.classes_)
    label_encoded = le.transform(df[label])
    return label_encoded

def plot_fig(df, save=False):    
    fig=sns.PairGrid(df, diag_sharey=False)
    plt.subplots_adjust(top=0.9)
    fig.fig.suptitle("Distribution")
    fig.map_lower(sns.kdeplot, cmap="Blues_d")
    fig.map_upper(plt.scatter)
    fig.map_diag(sns.distplot)
    if save == True: fig.savefig("pairplot.png")

    #plot kde joitplot
    fig = plt.figure()
    g =sns.jointplot(x=df.columns.values[0] ,y=df.columns.values[1] ,kind="kde", data=df)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Kernel Distribution Estimation")
    if save == True: fig.savefig("joint_kde_plot.png")
        
    #calculate acf, pcf
    acf, pcf = calc_pacf(df)
    ma = calc_ma(df)
    
    #calcularw return
    ret = calc_return(df)
    
    for i in range(len(df.columns)):
        #plot distribution
        fig = plt.figure()
        sns.distplot(df.iloc[:,i])
        plt.title("Histgram - " + df.columns[i])
        if save == True: fig.savefig("dist_plot_" + df.columns[i] + ".png")

        #plot series
        fig = plt.figure()
        df.iloc[:,i].plot()
        #plot moving average
        ma[i].plot(c='r')
        plt.title("Row data & Moving average - " + df.columns[i])
        if save == True: fig.savefig("series_plot_" + df.columns[i] + ".png")


        #plot acf
        fig = plt.figure()
        plt.bar(range(len(acf[i])), acf[i], width = 0.3)
        plt.title("Auto Correlation Function - " + df.columns[i])
        if save == True: fig.savefig("acf_plot_" + df.columns[i]+".png")

        #plot pcf
        fig = plt.figure()
        plt.bar(range(len(pcf[i])), pcf[i], width = 0.3)
        plt.title("Partial Auto Correlation Function - " + df.columns[i])
        if save == True: fig.savefig("pcf_plot_" + df.columns[i] + ".png")
        
        #plot fft
        fig = plt.figure()
        f = calc_fft(df.iloc[:,i])
        plt.plot(f[1], f[0])
        plt.title("Fast Fourier Transform - " + df.columns[i])
        if save == True: fig.savefig("fft_plot_" + df.columns[i] + ".png")
        
        #plot return value_i - value_i2
        fig = plt.figure()
        ret[i].plot()
        plt.title("Return - " + df.columns[i])
        if save == True: fig.savefig("return_plot_" + df.columns[i] + ".png")


def save_importances(importances_: pd.DataFrame):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    plt.figure(figsize=(8, 12))
    sns.barplot(
        x='gain',
        y='feature',
        data=importances_.sort_values('mean_gain', ascending=False)[:300])
    plt.tight_layout()
    plt.savefig('importances.png')
            
if __name__ == "__main__":
    #TEST : make_data
    #df1 = make_data("5t",288, 3)
    #print(df1)
    
    #TEST : make_anomaly
    #df2 = make_anomaly(df1, 288, 3, "pos")
    #print(df2)
    
    #TEST : index_convert
    #df3 = index_convert(df1)
    #print(df3)
    
    #titanic data
    base_dir = "/kaggle/input/"
    df_titanic = pd.read_csv(base_dir + "titanic/train.csv")
    #print(df_titanic)
    
    #TEST : missing_value
    df_titanic1 = missing_value(df_titanic,method="drop")
#     print(df_titanic1)
    
    # TEST : label_enc
    lb_list = ["Sex","Ticket","Embarked"]
    df_titanic2 = df_titanic1
    for lb in lb_list:
        label_encoded = label_enc(df_titanic1, lb)
        df_titanic2[lb] = label_encoded
    print(df_titanic2)

