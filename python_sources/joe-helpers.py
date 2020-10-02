# %% [code]
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression 
from sklearn.feature_selection import f_regression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans

import os

sns.set()

####################################################################################################
# plotting wrapper 
####################################################################################################
class Plotting:
    #TODO: 
    #more common display values like alpha and line colors etc:
    #y_jitter=0.025, scatter_kws={"edgecolor":"w"}, line_kws={'color':'orange'}
    
    ####################################################################################################
    # c'tor
    ####################################################################################################
    def __init__(self, data, save_plots=True, plots_folder=None, size_inches=None):
        if type(data) is str:
            self.data = pd.read_csv(data)
        else:
            self.data = data

        #if self.plots_folder is set, any plots created with a title will be saved to disc
        #default folder is cwd\plots, custom folder can be passed in via constructor
        #set self.plots_folder to None if you want to stop saving plots later
        if save_plots == True:
            if plots_folder == None:
                self.plots_folder = os.path.join(os.getcwd(), "plots")
                
                #override for kaggle
                if os.getcwd() == "/kaggle/working":
                    self.plots_folder = os.getcwd()
            else:
                self.plots_folder = plots_folder
        else:
            self.plots_folder = None
            
        #tuple for custom plot size, set to (w,h) in inches
        self.size_inches = size_inches
        self.palette = "GnBu_d"
    
    ####################################################################################################
    # saving and scaling...
    ####################################################################################################    
    def _SavePlot(self, plot_type, title):
        if title is not None:
            if self.plots_folder is not None:
                fig = plt.gcf()
                fig.savefig(os.path.join(self.plots_folder, plot_type + "_" + title + ".png"))

    def _SetPlotSize(self):
        if self.size_inches is not None:
            fig = plt.gcf()
            fig.set_size_inches(self.size_inches)
        
    ####################################################################################################
    # cluster plot
    ####################################################################################################
    def ClusterPlot(self, k, x, y, title=None, cluster_cols=None):
        df_copy = self.data.copy()
        km = KMeans(k)

        if cluster_cols is None:
            cluster_cols = [x, y]

        df_copy["cluster"] = km.fit_predict(self.data[cluster_cols])

        ax = sns.scatterplot(data=df_copy, x=x, y=y, hue="cluster", legend=None, palette=self.palette)
        ax.set_title(title, fontsize=16)
        self._SetPlotSize()
        self._SavePlot("cluster", title)
        plt.show()
        
    ####################################################################################################
    # wcss elbow plot
    ####################################################################################################
    def ClusteringElbowPlot(self, title=None, range_max=None):
        wcss=[]

        if range_max is None:
            rng = range(1, 11)
        else:
            rng = range(1, range_max)

        for i in rng:
            km = KMeans(i)
            km.fit(self.data)
            wcss.append(km.inertia_)

        sns.lineplot(rng, wcss)
        
        plt.title(title, fontsize=16)
        plt.xlabel("Number of clusters")
        plt.ylabel("Within-cluster Sum of Squares")
        
        self._SetPlotSize()
        self._SavePlot("elbow", title)
        plt.show()
    
    ####################################################################################################
    # distribution plots
    ####################################################################################################
    def DistPlot(self, colname, title=None):
        sns.distplot(self.data[colname])
        if title is not None:
            plt.title(title, fontsize=16)
            
        self._SetPlotSize()
        self._SavePlot("dist", title)
        plt.show()
        
    def BernoulliPlot(self, colname, title=None, tick_locations=None, tick_labels=None):
        sns.distplot(self.data[colname], kde=False, bins=[0,1,2])
        if title is not None:
            plt.title(title, fontsize=16)
        
        if tick_locations is not None:
            plt.xticks(tick_locations, tick_labels)

        self._SetPlotSize()
        self._SavePlot("bernoulli", title)
        plt.show()
        
    def CountPlot(self, colname, title=None, tick_locations=None, tick_labels=None, **kwargs):
        sns.countplot(data=self.data, x=colname, palette=self.palette, **kwargs)
        
        if title is not None:
            plt.title(title, fontsize=16)
        
        if tick_locations is not None:
            plt.xticks(tick_locations, tick_labels)

        self._SetPlotSize()
        self._SavePlot("count", title)
        plt.show()
        
    def BoxPlot(self, x, y, title=None, **kwargs):
        sns.boxplot(data=self.data, x=x, y=y, palette=self.palette, **kwargs)
        
        if title is not None:
            plt.title(title, fontsize=16)
        
        self._SetPlotSize()
        self._SavePlot("box", title)
        plt.show()


    ####################################################################################################
    # scatter plot
    ####################################################################################################
    def Scatter(self, x, y, title=None):
        sns.scatterplot(data=self.data, x=x, y=y)
        plt.xlabel(x)
        plt.ylabel(y)
        
        if title is not None:
            plt.title(title, fontsize=16)
        
        self._SetPlotSize()
        self._SavePlot("scatter", title)
        plt.show()
        
    ####################################################################################################
    # line plot
    ####################################################################################################
    def Lineplot(self, x, y, title=None):
        sns.lineplot(self.data[x], self.data[y])
        plt.xlabel(x)
        plt.ylabel(y)
        
        if title is not None:
            plt.title(title, fontsize=16)
        
        self._SetPlotSize()
        self._SavePlot("line", title)
        plt.show()
                
    ####################################################################################################
    # pair grid
    ####################################################################################################
    def PairGrid(self, title=None, exclude=None):
        #TODO: add a col wrap by splitting into seperate runs?
        #is there a workaround out there for this yet?
        df_copy = self.data.copy()
        values = self.data.columns.tolist()

        for col in self.data.columns:
            if np.issubdtype(self.data.dtypes[self.data.columns.get_loc(col)], np.number) == False:
                df_copy.drop(col, axis=1, inplace=True)

        if exclude is not None:
            for col in exclude:
                df_copy.drop(col, axis=1, inplace=True)

        g = sns.PairGrid(df_copy)
        g = g.map_diag(plt.hist, edgecolor="w")
        g = g.map_offdiag(plt.scatter, edgecolor="w")

        if title is not None:
            g.fig.suptitle(title, fontsize=16)
            g.fig.subplots_adjust(top=0.95)
            
        self._SetPlotSize()
        self._SavePlot("pgrd", title)
        plt.show()
    
    ####################################################################################################
    # internal pairplot function
    ####################################################################################################
    def _DrawAllAgainst(self, y, func, exclude_cols, title, **kwargs):
        values = self.data.columns.tolist()
        values.remove(y)

        for col in self.data.columns:
            if np.issubdtype(self.data.dtypes[self.data.columns.get_loc(col)], np.number) == False:
                values.remove(col) #drop non-numeric cols from the plots

        if exclude_cols is not None:
            for col in exclude_cols:
                values.remove(col)

        g = sns.PairGrid(self.data, y_vars=y, x_vars=values)        
        g.map(func, **kwargs)

        if title is not None:
            g.fig.suptitle(title, fontsize=16)
            g.fig.subplots_adjust(top=0.90)

    ####################################################################################################
    # pairplot functions
    ####################################################################################################
    def ScatterAllAgainst(self, y, title=None, exclude_cols=None):
        self._DrawAllAgainst(y, sns.scatterplot, exclude_cols, title)
        self._SavePlot("scatter_all", title)
        plt.show()

    def PlotAllAgainst(self, y, title=None, exclude_cols=None):
        self._DrawAllAgainst(y, sns.lineplot, exclude_cols, title)
        self._SavePlot("line_all", title)
        plt.show()

    def RegPlotAllAgainst(self, y, title=None, exclude_cols=None):
        self._DrawAllAgainst(y, sns.regplot, exclude_cols, title, line_kws={'color':'orange'})
        self._SavePlot("regr_all", title)
        plt.show()

    def LogisticPlotAllAgainst(self, y, title=None, exclude_cols=None):
        self._DrawAllAgainst(y, sns.regplot, exclude_cols, title, logistic=True, ci=None, line_kws={'color':'orange'})
        self._SavePlot("logit_all", title)
        plt.show()
        
    def FacetGrid(self, x, y, col, func, title=None, **kwargs):
    ####################################################################################################
    # facetgrid
    ####################################################################################################
        g = sns.FacetGrid(data=self.data, col=col)

        if (x is not None) and (y is not None):
            g.map(func, x, y, **kwargs)
        elif (x is not None):    
            g.map(func, x, **kwargs)
        elif (y is not None):
            g.map(func, y, **kwargs)

        if title is not None:
            plt.subplots_adjust(top=0.8)
            g.fig.suptitle(title)

        self._SavePlot("fgrd", title)
        plt.show()

    ####################################################################################################
    # regression plots
    ####################################################################################################
    def RegressionPlot(self, x, y, title=None, logistic=False, ci=95):    
        ax = sns.regplot(x=x, y=y, data=self.data, logistic=logistic, 
                         y_jitter=0.025, scatter_kws={'alpha':0.2}, line_kws={'color':'orange'}, ci=ci)
        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        if title is not None:
            plt.title(title, fontsize=16)
            
        self._SetPlotSize()
        
        if logistic == False:
            self._SavePlot("regr", title)
        else:
            self._SavePlot("logit", title)
            
        plt.show()
        
####################################################################################################
# data cleaning helper functions
####################################################################################################
class DataCleaner:
    def __init__(self, data, copy_data=False):
        if type(data) is str:
            self.data = pd.read_csv(data)
        else:
            if copy_data is True:
                self.data = data.copy()
            else:
                self.data = data
                    
    def _PrintHeading(self, text):
            print("")
            print(text)
            print("------------------------------")
    
    def _FmtPcnt(self, num, den, rnd=1):
        return str(round((num / den * 100), rnd)) + "%"
    
    def Concat(self, data):
        if type(data) is str:
            df2 = pd.read_csv(data)
        else:
            df2 = data
            
        self.data = pd.concat([self.data, df2], ignore_index=True)
        self.ResetIndex()
    
    def DescribeAll(self):
        self.Describe(include="all")
        
    def Describe(self, col=None, include=None):
            self._PrintHeading("Description:")
            
            if col is not None:
            #SINGLE COL, shorter version...
                warnings = ""
                isnumeric = np.issubdtype(self.data[col].dtype, np.number)
                descr = self.data[col].describe()
                print(descr)
                
                count = self.data[col].isnull().count()
                isnull = self.data[col].isnull().sum()
                
                if isnull > 0:
                    warnings += str(isnull) + " rows are null (" + self._FmtPcnt(isnull, count) + ")"
                  
                if isnumeric:
                    if self.data[col].skew() > 1:
                        warnings += "Skew is " + str(self.data[col].skew())
                
                if len(warnings) > 0:
                    self._PrintHeading("Warnings:")
                    print(warnings)
                    
                return
                
            #MULTIPLE COLS
            descr = self.data.describe(include=include)
            
            #add skew row (this is a bit disgusting, there must be a more direct way of doing this)
            skew = self.data.skew()
            dct = {}
            for c in descr.columns:
                dct[c] = 0
            for c in skew.index:
                dct[c] = skew[c]

            sr = pd.Series(dct)
            sr = sr.rename("skew")
            descr = descr.append(sr)
            
            #work out the nulls
            counts = descr.loc["count"]
            rows = len(self.data)
            pcnts = {}

            for c in counts.index:
                if counts[c] < rows:
                    pcnts[c] = self._FmtPcnt(rows - counts[c], rows)
                else:
                    pcnts[c] = 0

            #add null rows
            sr = pd.Series(pcnts)
            sr = sr.rename("null")
            descr = descr.append(sr)
                        
            #print descr now we've added our own rows
            print(descr)
            
    def PrintUniqueValues(self, col):
        self._PrintHeading("Unique values for " + str(col) + ":")
        print(np.sort(self.data[col].unique()))

    def PrintValueCounts(self, col_name=None):
        if col_name is None:
            self._PrintHeading("Value counts:")

            for col in self.data.columns:
                print(self.data[col].value_counts())
                print("------------------------------")
        else:
            self._PrintHeading("Value counts for " + str(col_name) + ":")
            print(self.data[col_name].value_counts())

    def DropColumns(self, cols):
        self.data.drop(cols, axis=1, inplace=True)
        
    def AddColumn(self, col, fill_value):
        self.data.insert(len(self.data.columns), col, fill_value)

    def DropRowsMissingValues(self, col=None):
        if col is None:
            self.data = self.data.dropna(axis=0)
        else:
            self.data = self.data[col].dropna(axis=0)
        
    def SetMissingValues(self, col, val):
        self.data[col] = self.data[col].fillna(val)
        
    def SetMissingValuesToMedian(self, col):
        med = self.data[col].median()
        self.data[col] = self.data[col].fillna(med)
        
    def SetMissingValuesToMean(self, col):
        mean = self.data[col].mean()
        self.data[col] = self.data[col].fillna(mean)

    def Top(self, colname):
        q = self.data[colname].quantile(0.99)
        self.data = self.data[self.data[colname] < q]

    def Tail(self, colname):
        q = self.data[colname].quantile(0.01)
        self.data = self.data[self.data[colname] > q]

    def TopAndTail(self, colname):
        q99 = self.data[colname].quantile(0.99)
        q01 = self.data[colname].quantile(0.01)
        df = self._ata[self.data[colname] > q01]
        df = df[df[colname] < q99]
        self.data = df

    def ResetIndex(self):
        self.data.reset_index(drop=True)

    def Log(self, col, drop_original=True):
        log_data = np.log(self.data[col])
        self.data["log_" + col] = log_data

        if drop_original == True:
            self.DropColumns(col)
            
    def Exp(self, col, drop_original=True):
        #if col is named log_Variable then new col is called Variable, otherwise exp_Variable 
        if col.lower()[0:4] == "log_":
            name = col[4:]
        else:
            name = "exp_" + col
            
        exp_data = np.exp(self.data[col])
        self.data[name] = exp_data

        if drop_original == True:
            self.DropColumns(col)
            
    def Standardize(self, col, drop_original=True):
        std_data = preprocessing.scale(self.data[col])
        self.data["std_" + col] = std_data

        if drop_original == True:
            self.DropColumns(col)

    def AddDummies(self, drop=True):
        self.data = pd.get_dummies(self.data, drop_first=drop)

    def MoveToStart(self, col):
        cols = self.data.columns.values.tolist()
        old_first = cols[0]
        old_col_pos = cols.index(col)
        cols[old_col_pos] = old_first
        cols[0] = col
        self.data = self.data[cols]
        
    def Remap(self, col, dict_values):
        self.data[col] = self.data[col].map(dict_values)
        
    def AddConstant(self):
        self.data = sm.add_constant(self.data)    
        
####################################################################################################
# regression helper functions
####################################################################################################
class Regressions:
    
    ####################################################################################################
    # VIF info
    ####################################################################################################    
    def VIF(data, cols_array):        
        variables = data[cols_array]
        vif = pd.DataFrame()
        vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
        vif["features"] = variables.columns
        return vif

    ####################################################################################################
    # calculates an adjusted r2 score
    ####################################################################################################
    def AdjustedR2(x, y):
        regr = LinearRegression()
        regr.fit(x, y)
        r2 = regr.score(x, y)    
        n = x.shape[0]
        p = x.shape[1]

        return 1-(1-r2)*(n-1)/(n-p-1)

    ####################################################################################################
    # prints a regression summary table
    ####################################################################################################
    def LinearRegrSummary(x, y, standardize=False, input_columns=None):
        regr = LinearRegression()
        
        if input_columns is None:
            xcols = x.columns
        else:
            xcols = input_columns
        
        if standardize == True:
            scaler = StandardScaler()
            scaler.fit(x)
            scaled_x = scaler.transform(x)
            regr.fit(scaled_x, y)
            r2 = regr.score(scaled_x, y)
        else:        
            regr.fit(x, y)
            r2 = regr.score(x, y)

        n = x.shape[0]
        p = x.shape[1]
        adjR2 = 1-(1-r2)*(n-1)/(n-p-1)

        dct = {}
        dct["R2"] = round(r2, 3)
        dct["Adjusted R2"] = round(adjR2, 3)
        dct["Intercept"] = round(regr.intercept_, 3)
        s1 = pd.DataFrame(dct.items(), columns=["Vars", "Values"])
        print(s1)
        print("")

        fregr = f_regression(x, y)        
        f_stats = fregr[0].round(3)
        p_values = fregr[1].round(3)

        s2 = pd.DataFrame(data = xcols.values, columns=["Features"])        
        s2["Coefs"] = regr.coef_.round(3)
        s2["F-Stats"] = f_stats
        s2["P-Values"] = p_values
        print(s2)
        print("")

        i = 0
        while i < len(xcols):
            if p_values[i] > 0.05:
                print("*** Warning: '" + xcols[i] + "' P-Value is high (" + str(round(p_values[i], 3)) + ")")
            i += 1    
            
    ####################################################################################################
    # creates a confusion matrix for a logistic regression
    # returns: matrix, accuracy, misclassification
    ####################################################################################################
    def ConfusionMatrix(x, y, model):
        pred_values = model.predict(x) #predict using logit model
        bins = np.array([0,0.5,1]) #specify the bins 
        cm = np.histogram2d(y, pred_values, bins=bins)[0] #create the confusion matrix through a histogram
        #all because you can't create a cm from predicted data... :-/
        
        confusion_matrix = pd.DataFrame(cm)
        confusion_matrix.columns = ["Predicted 0", "Predicted 1"]
        confusion_matrix = confusion_matrix.rename(index={0:"Was 0", 1:"Was 1"})
        print(confusion_matrix)
        
        ac = ((cm[0,0] + cm[1,1]) / cm.sum() * 100).round(2)
        mc = ((cm[0,1] + cm[1,0]) / cm.sum() * 100).round(2)
        
        print("")
        print("Accuracy:   " + str(ac) + "%")
        print("Misclassification: " + str(mc) + "%")
        
        return cm, ac, mc # return the confusion matrix, accuracy and misclassification rate