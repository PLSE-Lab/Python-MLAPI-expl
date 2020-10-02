# skmem.MemReducer
# Smart memory reduction for pandas.

# A transformer to quickly reduce dataframe memory by converting memory-hungry
# dtypes to ones needing less memory. Advantages include:
#     - Fully compatible with scikit-learn. Easy to integrate with other 
#       transformers and pipelines.
#     - Preserves data integrity. Set simple parameters to control
#       treatment of floats and objects.
#     - Easy to customize. Use class inheritance or directly change modular
#       functions as needed.
#     - Efficient. Save time with vectorized functions that process data
#       faster than most parallelized solutions.

# The latest version includes transformation to pandas nullable integer types. 
# By default, pandas read functions will cast columns as float64 if there are null
# values, even if the values present are integers. The new data types Int64,
# Int32, etc. can be used in columns with null values to save memory. Transforming
# columns to nullable types is optional since some packages don't handle them.

# The main() function below has examples using Kaggle data.


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import validation


class MemReducer(BaseEstimator, TransformerMixin):
    """ Converts dataframe columns to dtypes requiring less memory. Returns a
    dataframe with memory-efficient dtypes where possible.

    Integers, 64-bit floats and objects/strings can be converted.
    Parameters provide control for treatment of floats and objects.

    Parameters
    ___________
    max_unique_pct : float, optional, default=0.5
        Sets maximum threshold for converting object columns to categoricals.
        Threshold is compared to the number of unique values as a percent of
        column length. 0.0 prevents all conversions and 1.0 allows all
        conversions.

    nullables: boolean, optional, default=True
        Specifies whether to convert float columns with all whole numbers and
        null values into pandas nullable integers.

    Example
    --------
    >>> import skmem
    >>> df = pd.DataFrame({'cat': np.tile(['a', 'b'], 500_000),
                'true_int': np.tile(np.arange(-5, 5), 100_000),
                'float': np.arange(0., 1_000_000.),
                'nullable': np.tile([np.nan, 1, 1, 2, 3], 200_000)
                })

    >>> print(df.dtypes)
    |cat     object
    |true_int    int64
    |float    float64
    |nullable    float64
    |dtype: object
    
    >>> mr = skmem.MemReducer(max_unique_pct=0.8)
    >>> df_small = mr.fit_transform(df, float_cols=['float'])
    |Memory in: 0.08 GB
    |Starting integers.
    |Starting floats.
    |Starting objects.
    |Memory out: 0.01 GB
    |Reduction: 92.7%
    
    >>> print(df_small.dtypes)
    |cat      category
    |true_int    int8
    |float     float32
    |nullable    int8
    |dtype: object

    Notes
    -----
    Downcasting to float dtypes below 32-bits (np.float16, np.float8)
    is not supported.
    """

    
    def __init__(self, max_unique_pct=0.2, nullables=True):
        self.max_unique_pct = max_unique_pct
        self.nullables = nullables
        
    def fit(self, df, float_cols=None):
        """ Identify dataframe and any float columns to be reduced.

        Parameters
        ----------
        df : pandas DataFrame
            The dataframe used as the basis for conversion.

        float_cols : list, optional, default=None
            A list of column names to be converted from np.float64 to
            np.float32.
        """
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"'{type(df).__name__}' object is not a pandas \
                    dataframe.")
        
        self.float_candidates = float_cols
        return self

    
    # Helper functions for .transform()
    def reduce_ints(self, df):
        int_cols = df.select_dtypes('integer').columns
        if not int_cols.empty:
            print("Starting integers.", flush=True)
            mins = df[int_cols].min()
            unsigneds = mins.index[mins >= 0]
            df[unsigneds] = df[unsigneds].apply(pd.to_numeric,
                                                downcast='unsigned')
            signeds = mins.index[mins < 0]
            df[signeds] = df[signeds].apply(pd.to_numeric,
                                            downcast='signed')
        return df

    
    def reduce_floats(self, df, float_cols):
        print("Starting floats.", flush=True)
        
        if not isinstance(float_cols, list):
            print(f"'{type(float_cols).__name__}' object is not a list,\
                    skipping floats.")
        else:
            true_float_cols = df.select_dtypes(np.float64).columns.tolist()
            non_float64s = [f for f in float_cols if f not in true_float_cols]
            if non_float64s:
                print("Skipping columns that are not np.float64")

            convertibles = [f for f in float_cols if f in true_float_cols]
            if convertibles:
                df[convertibles] = df[convertibles].astype(np.float32)
        return df
    
    
    def reduce_objs(self, df, max_pct):   
        if not 0<=max_pct<=1:
            raise ValueError("max_unique_pct must be between 0 and 1")

        obj_cols = df.select_dtypes('object').columns
        if not obj_cols.empty:
            print("Starting objects.", flush=True)
            for oc in obj_cols:
                try:
                    df[oc] = pd.to_numeric(df[oc], downcast='integer')
                except:
                    pass
                else: 
                    print(f"Converting {oc} to numbers.")
                    
        new_obj_cols = df.select_dtypes('object').columns    
        if not new_obj_cols.empty:
            category_mask = df[new_obj_cols].nunique().to_numpy()/len(df) <= max_pct
            cat_cols = new_obj_cols[category_mask]
            if not cat_cols.empty:
                df[cat_cols] = df[cat_cols].astype('category')
                print(f"Converting {cat_cols.tolist()} to categories.")
        return df
    
    
    def reduce_nullables(self, df):
        print("Starting nullables.", flush=True)
        
        true_float_cols = df.select_dtypes('float').columns
        remainders = df[true_float_cols].mod(1).max(axis=0)
        nulls = df[true_float_cols].isnull().sum()
        
        convertibles = remainders[remainders==0].index \
                        .intersection(nulls[nulls!=0].index) \
                        .tolist()
        if convertibles:
            start_types = df[convertibles].dtypes
            df[convertibles] = df[convertibles].convert_dtypes()
            end_types = df[convertibles].dtypes

            changed = end_types[~end_types.eq(start_types)]                     
            changed_nums = changed[end_types!='string'].index
                        
            #TODO: change ifs and loops to np.arrays or similar
            #       and add in unsigned ints
            
            for cc in changed_nums:  
                max_int = df[cc].abs().max()
                if 32767 < max_int <=  2147483647:
                    df[cc] = df[cc].astype('Int32')
                elif 127 < max_int <= 32767:
                    df[cc] = df[cc].astype('Int16')
                elif max_int <= 127:
                    df[cc] = df[cc].astype('Int8')
                    
            print(f"Columns converted to nullables: {changed.index.tolist()}")
        
        else:
            print("No candidates for nullable integers.")

        return df

    
    def transform(self, df):
        """ Convert dataframe columns to dtypes requiring lower memory.

        Parameters
        ----------
        df : pandas DataFrame
            The dataframe to be converted.
        """

        validation.check_is_fitted(self, 'float_candidates')

        print("Getting memory usage.")
        memory_MB_in = df.memory_usage(deep=True).sum()/(1024**2)
        print(f"\nMemory in: {memory_MB_in:.2f} MB")

        df = self.reduce_ints(df)
        if self.float_candidates:
            df = self.reduce_floats(df, self.float_candidates)
        df = self.reduce_objs(df, self.max_unique_pct)
        if self.nullables:
            df = self.reduce_nullables(df)

        memory_MB_out = df.memory_usage(deep=True).sum()/(1024**2)
        print(f"Memory out: {memory_MB_out:.2f} MB",
              f"Reduction: {1 - memory_MB_out/memory_MB_in:.1%}\n")

        return df

    
def main():
    """skmem is designed to be a utility script. This function helps with testing
    and demonstrates how it works when run directly.
    """
    
    # Predict Future Sales
    df1 = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
    mr = MemReducer(max_unique_pct=0.1)  # set threshold for converting to categoricals
    df1_small = mr.fit_transform(df1, float_cols=['item_price', 'item_cnt_day'])  # convert to float32
    
    
    # Restaurant Recommendation Challenge
    print("Reading 2.9GB dataframe.")
    df2 = pd.read_csv('../input/restaurant-recommendation-challenge/train_full.csv', low_memory=False)
    mr = MemReducer(max_unique_pct=0.2, nullables=True)  # explicitly set optional parameters
    df2_small = mr.fit_transform(df2, float_cols=['latitude_x', 'longitude_x'])
    

if __name__=="__main__":
    main()