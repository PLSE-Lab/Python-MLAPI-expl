"""Functions to interactively cut the data into buckets and plot the results"""

# Useful to have a version number for checking updates
__version__ = '0.1.0'

############
# Contents #
############
# - Setup
# - Assign buckets
# - Group and aggregate
# - Set coordinates
# - Pipeline functions
# - Plotting
# - Running interactively

#########
# Setup #
#########
# Import built-in modules
import functools
import inspect

# Import external modules
import numpy as np
import pandas as pd
import bokeh
import bokeh.palettes

##################
# Assign buckets #
##################
def divide_n(df, bucket_var, n_bins=10):
    """
    Assign each row of `df` to a bucket by dividing the range of the 
    `bucket_var` column into `n_bins` number of equal width intervals.
    
    df: DataFrame
    bucket_var: Name of the column of df to use for dividing.
    n_bins: positive integer number of buckets.
    
    Returns: df with the additional `bucket` column 
        The `bucket` column is Categorical data type consisting of Intervals
        that partition the interval from just below min(bucket_var) to 
        max(bucket_var).
    """
    df_w_buckets = df.assign(
        bucket=lambda df: pd.cut(df[bucket_var], bins=n_bins)
    )
    return(df_w_buckets)


def custom_width(df, bucket_var, width, boundary=0, first_break=None, last_break=None):
    """
    Assign each row of `df` to a bucket by dividing the range of the 
    `bucket_var` column into `n_bins` number of equal width intervals.
    
    df: DataFrame
    bucket_var: Name of the column of df to use for dividing.
    width: Positive width of the buckets
    boundary: Edge of one of the buckets, if the data extended that far
    first_break: All values below this (if any) are grouped into one bucket
    last_break: All values above this (if any) are grouped into one bucket
    
    Returns: df with the additional `bucket` column 
        The `bucket` column is Categorical data type consisting of Intervals
        that partition the interval from just below min(bucket_var) to 
        max(bucket_var).
    """
    var_min, var_max = df[bucket_var].min(), df[bucket_var].max()
    extended_min = var_min - 0.001 * np.min([(var_max - var_min), width])

    # Set bucket edges
    start = np.floor((extended_min - boundary) / width) * width + boundary
    stop = np.ceil((var_max - boundary) / width) * width + boundary
    num = int((stop - start) / width) + 1
    breaks_all = np.array([
        extended_min,
        *np.linspace(start, stop, num)[1:-1],
        var_max,
    ])
    
    # Clip lower and upper buckets
    breaks_clipped = breaks_all
    if first_break is not None or last_break is not None:
        breaks_clipped = np.unique(np.array([
            breaks_all.min(),
            *np.clip(breaks_all, first_break, last_break),
            breaks_all.max(),
        ]))
    breaks_clipped
    
    df_w_buckets = df.assign(
        bucket=lambda df: pd.cut(df[bucket_var], bins=breaks_clipped)
    )
    return(df_w_buckets)


def weighted_quantiles(df, bucket_var, n_bins=10, bucket_wgt=None, validate=True):
    """
    Assign each row of `df` to a bucket by splitting column `bucket_var`
    into `n_bins` weighted quantiles, weighted by `bucket_wgt`.
    
    bucket_var: Column name of the values to find the quantiles.
        Must not be constant (i.e. just one value for all rows).
    n_bins: Target number of quantiles, but could end up with fewer because
        there are only a finite number of potential cut points.
    bucket_wgt: Weights to use to calculate the weighted quantiles.
        If None (default) or 'const' then equal weights are used for all rows.
        Must be non-negative with at least one postive value.
    validate: boolean. Set to False to omit validation checks on inputs.
    
    Returns: df with the additional `bucket` column 
        The `bucket` column is Categorical data type consisting of Intervals
        that partition the interval from 0 to sum(bucket_wgt).
    """
    if bucket_wgt is None:
        bucket_wgt = 'const'
    if bucket_wgt == 'const':
        df = df.assign(const = 1)
    
    if validate:
        if df[bucket_var].nunique() == 1:
            raise ValueError(
                f"weighted_quantiles: bucket_var column '{bucket_var}' "
                "must not be constant"
            )
        if (df[bucket_wgt] < 0).any() or (df[bucket_wgt] == 0).all():
            raise ValueError(
                f"weighted_quantiles: bucket_wgt column '{bucket_wgt}' "
                "must be non-negative with at least one strictly positive value"
            )
    
    res = df.sort_values(bucket_var).assign(
        **{col + '_cum_rows': lambda df: (
            df[bucket_wgt].cumsum()
        ) for col in [bucket_wgt]},
        # Ensure that the quantiles cannot split rows with the same value of bucket_var
        **{col + '_cum': lambda df: (
            df.groupby(bucket_var)[col + '_cum_rows'].transform('max')
        ) for col in [bucket_wgt]},
        bucket=lambda df: pd.qcut(df[bucket_wgt + '_cum'], q=n_bins, duplicates='drop'),
    )
    return(res)


def all_levels(df, bucket_var, include_levels=None, ret_map=False):
    """
    Assign each row of `df` to a bucket according to the unique 
    values of `bucket_var`.
    
    bucket_var: Column name of the values to split on.
        Missing values will not be assigned to an interval.
    include_levels: Level values to guarantee to include 
        even if they do not appear in the values of bucket_var.
        Missing values are ignored.
    
    Returns: 
        df with the additional `bucket` column
            The `bucket` column is Categorical data type consisting of 
            Intervals that partition a range, plus possible NaN.
        If ret_map is True, also return a Series mapping bucket values
            to bucket intervals.
    """
    # Format inputs
    if include_levels is not None:
        if not isinstance(include_levels, pd.Series):
            include_levels = pd.Series(include_levels)
    
    # Get the mapping from level value to an appropriate interval
    buckets_vals = pd.concat([
        df[bucket_var], include_levels
    ]).drop_duplicates().sort_values(
    ).reset_index(drop=True).dropna().to_frame('val')
    
    # Add a column of intervals (there may be some intervals with no rows)
    if np.issubdtype(df[bucket_var].dtype, np.number):
        # If the values are numeric then take the smallest width
        min_diff = np.min(np.diff(buckets_vals['val']))
        buckets_map = buckets_vals.assign(
            interval=lambda df: pd.cut(df['val'], pd.interval_range(
                start=df['val'].min() - min_diff/2,
                end=df['val'].max() + min_diff/2,
                freq=min_diff
            ))
        )
    else:
        buckets_map = buckets_vals.assign(
            interval=lambda df: pd.interval_range(start=0., periods=df.shape[0], freq=1.)
        )
    
    # Convert to a Series
    buckets_map = buckets_map.reset_index(drop=True)
    
    # Assign buckets and map to intervals
    res = df.assign(
        bucket=lambda df: df[bucket_var].astype(
            pd.CategoricalDtype(buckets_map['val']) 
        ).cat.rename_categories(
            buckets_map.set_index('val')['interval']
        )
    )
    
    if ret_map:
        return(res, buckets_map)
    return(res)

#######################
# Group and aggregate #
#######################
def group_and_agg(df_w_buckets, x_var, stat_wgt=None, stat_vars=None):
    """
    Group by bucket and calculate aggregate values in each bucket
    
    df_w_buckets: Result of an 'assign_buckets' function.
        i.e. a DataFrame with a `bucket` column the is Categorical
        with Interval categories that partition a range.
        Rows with missing `bucket` value are excluded from the grouping.
    x_var: Column name of variable that will be plotted on the x axis.
    stat_wgt: Weights for the weighted distributions of stat_vars.
        If None (default) or 'const' then equal weights are used for all rows.
        Must be non-negative with at least one postive value.
    stat_vars: 
        If None (default) or empty list, no values are calculated.
    
    Returns: Aggregated DataFrame for plotting.
    """
    # Set defaults
    if stat_wgt is None:
        stat_wgt = 'const'
    if stat_wgt == 'const':
        df_w_buckets = df_w_buckets.assign(const = 1)
    if stat_vars is None:
        stat_vars = []
    
    # Format inputs and defaults
    if not isinstance(stat_vars, list):
        stat_vars = [stat_vars]
    
    # Variables for which we want the (weighted) distribution in each bucket
    agg_vars_all = stat_vars
    if np.issubdtype(df_w_buckets[x_var].dtype, np.number):
        agg_vars_all = [x_var] + agg_vars_all
    # Ensure they are unique (and maintain order)
    agg_vars = pd.Series(agg_vars_all).drop_duplicates()
    
    df_agg = df_w_buckets.assign(
        **{col + '_x_wgt': (
            lambda df, col=col: df[col] * df[stat_wgt]
        ) for col in agg_vars},
    ).groupby(
        # Group by the buckets
        'bucket', sort=False
    ).agg(
        # Aggregate calculation for rows in each bucket
        n_obs=('bucket', 'size'),  # It is possible that a bucket contains zero rows
        **{col: (col, 'sum') for col in [stat_wgt]},
        **{stat_var + '_wgt_sum': (
            stat_var + '_x_wgt', 'sum'
        ) for stat_var in agg_vars},
        x_min=(x_var, 'min'),
        x_max=(x_var, 'max'),
    ).pipe(
        # Convert the index to an IntervalIndex
        lambda df: df.set_index(df.index.categories)
    ).sort_index().assign(
        # Calculate the weighted average of the stats
        **{stat_var + '_wgt_av': (
            lambda df, stat_var=stat_var: df[stat_var + '_wgt_sum'] / df[stat_wgt]
        ) for stat_var in agg_vars},
    )
    
    return(df_agg)


###################
# Set coordinates #
###################
# Functions to set the x-axis edges `x_left` and `x_right`
def x_edges_min_max(df_agg):
    """
    Set the x-axis edges to be the min and max values of `x_var`.
    Does not make sense to use this option when min and max are not numeric.
    This might result in zero width intervals, in which case a warning is given.
    """
    if not np.issubdtype(df_agg['x_min'].dtype, np.number):
        raise ValueError(
            "\n\tx_edges_min_max: This method can only be used when"
            "\n\tx_min and x_max are numeric data types."
        )
        
    if (df_agg['x_min'] == df['x_max']).any():
        warning(
            "x_edges_min_max: At least one bucket has x_min == x_max, "
            "so using this method will result in zero width intervals."
        )
    
    res = df_agg.assign(
        # Get the coordinates for plot: interval edges
        x_left=lambda df: df['x_min'],
        x_right=lambda df: df['x_max'],
    )
    return(res)


def x_edges_interval(df_agg):
    """Set the x-axis edges to be the edges of the bucket interval"""
    res = df_agg.assign(
        x_left=lambda df: df.index.left,
        x_right=lambda df: df.index.right,
    )
    return(res)


def x_edges_unit(df_agg):
    """
    Set the x-axis edges to be the edges of equally spaced intervals
    of width 1.
    """
    res = df_agg.assign(
        interval=lambda df: pd.interval_range(start=0., periods=df.shape[0], freq=1.),
        x_left=lambda df: pd.IntervalIndex(df['interval']).left,
        x_right=lambda df: pd.IntervalIndex(df['interval']).right,
    ).drop(columns='interval')
    return(res)


# Functions to set the x-axis point
def x_point_mid(df_agg):
    """Set the x_point to be mid-way between x_left and x_right"""
    res = df_agg.assign(
        x_point=lambda df: (df['x_left'] + df['x_right']) / 2.
    )
    return(res)

def x_point_wgt_av(df_agg, x_var):
    """
    Set the x_point to be the weighted average of x_var within the bucket,
    weighted by stat_wgt.
    """
    if not (x_var + '_wgt_av') in df_agg.columns:
        raise ValueError(
            "\n\tx_point_wgt_av: This method can only be used when"
            "\n\tthe weighted average has already been calculated."
        )
    
    res = df_agg.assign(
        x_point=lambda df: df[x_var + '_wgt_av']
    )
    return(res)

# Functions to set the x-axis labels
def x_label_none(df_agg):
    res = df_agg.copy()
    if 'x_label' in df_agg.columns:
        res = res.drop(columns='x_label')
    return(res)


######################
# Pipeline functions #
######################
# Constant to store pipeline functions
pipe_funcs_df = pd.DataFrame(
    columns=['task', 'func', 'alias'],
    data = [
        ('x_edges', x_edges_interval, ['interval']),
        ('x_edges', x_edges_min_max, ['min_max']),
        ('x_edges', x_edges_unit, ['unit']),
        ('x_point', x_point_mid, ['mid']),
        ('x_point', x_point_wgt_av, ['wgt_av']),
        ('x_label', x_label_none, ['none']),
    ],
).assign(
    name=lambda df: df['func'].apply(lambda f: f.__name__),
    arg_names=lambda df: df['func'].apply(
        lambda f: inspect.getfullargspec(f)[0][1:]
    ),
).set_index(['task', 'name'])


def get_pipeline_func(
    task, search_term,
    kwarg_keys=None, calling_func='',
    pipe_funcs_df=pipe_funcs_df
):
    """
    TODO: Write docstring <<<<<<<<<<<<<
    """
    # Set defaults
    if kwarg_keys is None:
        kwarg_keys = []
    
    # Find function row
    task_df = pipe_funcs_df.loc[task,:]
    func_row = task_df.loc[task_df.index == search_term, :]    
    if func_row.shape[0] != 1:
        func_row = task_df.loc[[search_term in ali for ali in task_df.alias], :]
    if func_row.shape[0] != 1:
        raise ValueError(
            f"\n\t{calling_func}: Cannot find '{search_term}' within the"
            f"\n\tavailable '{task}' pipeline functions."
        )
        
    # Check arguments are supplied
    for req_arg in func_row['arg_names'][0]:
        if not req_arg in kwarg_keys:
            raise ValueError(
                f"\n\t{calling_func}: To use the '{search_term}' as a '{task}' pipeline"
                f"\n\tfunction, you must specify '{req_arg}' as a keyword argument."
            )
    return(func_row['func'][0], func_row['arg_names'][0])


def add_x_coords(df_agg, x_edges=None, x_point=None, x_label=None, **kwargs):
    """
    Given a DataFrame where each row is a bucket, add x-axis 
    properties to be used for plotting. See pipe_funcs_df for 
    available options.
    
    x_edges: How to position the x-axis edges.
        Default: 'interval'
    x_point: Where to position each bucket point on the x-axis.
        Default: 'mid'
    x_label: Option for x-axis label.
        Default: 'none'
    **kwargs: Additional arguments to pass to the functions.
    """
    # Set variables for use throughout the function
    calling_func = 'add_x_coords'
    kwarg_keys = list(kwargs.keys())
    
    # Set defaults
    if x_edges is None:
        x_edges = 'interval'
    if x_point is None:
        x_point = 'mid'
    if x_label is None:
        x_label = 'none'
    
    # Get pipeline functions
    x_edges_func = [
        functools.partial(full_func, **{arg_name: kwargs[arg_name] for arg_name in arg_names})
        for full_func, arg_names in [get_pipeline_func('x_edges', x_edges, kwarg_keys, calling_func)]
    ][0]
    x_point_func = [
        functools.partial(full_func, **{arg_name: kwargs[arg_name] for arg_name in arg_names})
        for full_func, arg_names in [get_pipeline_func('x_point', x_point, kwarg_keys, calling_func)]
    ][0]
    x_label_func = [
        functools.partial(full_func, **{arg_name: kwargs[arg_name] for arg_name in arg_names})
        for full_func, arg_names in [get_pipeline_func('x_label', x_label, kwarg_keys, calling_func)]
    ][0]
    
    # Apply the functions
    res = df_agg.pipe(
        lambda df: x_edges_func(df)
    ).pipe(
        lambda df: x_point_func(df)
    ).pipe(
        lambda df: x_label_func(df)
    )
    return(res)


############
# Plotting #
############
def expand_lims(df, pct_buffer_below=0.05, pct_buffer_above=0.05, include_vals=None):
    """
    Find the range over all columns of df. Then expand these 
    below and above by a percentage of the total range.
    
    df: Consider all values in all columns
    include_vals: Additional values to consider
    
    Returns: Series with rows 'start' and 'end' of the expanded range
    """
    # If a Series is passed, convert it to a DataFrame
    try:
        df = df.to_frame()
    except:
        pass
    # Case where df has no columns, just fill in default vals
    if df.shape[1] == 0:
        res_range = pd.Series({'start': 0, 'end': 1})
        return(res_range)
    if include_vals is None:
        include_vals = []
    if not isinstance(include_vals, list):
        include_vals = [include_vals]
    
    res_range = pd.concat([
        df.reset_index(drop=True),
        # Add a column of extra values to the DataFrame to take these into account
        pd.DataFrame({'_extra_vals': include_vals}),
    ], axis=1).apply(
        # Get the range (min and max) over the DataFrame
        ['min', 'max']).agg({'min': 'min', 'max': 'max'}, axis=1).agg({
        # Expanded range
        'start': lambda c: c['max'] - (1 + pct_buffer_below) * (c['max'] - c['min']),
        'end': lambda c: c['min'] + (1 + pct_buffer_above) * (c['max'] - c['min']),
    })
    return(res_range)


def create_bplot(
    df_for_plt, stat_wgt, stat_vars,
    cols=bokeh.palettes.Dark2[8],
):
    """Create bucket plot object from aggregated data"""
    # Set up the figure
    bkp = bokeh.plotting.figure(
        title="One-way plot", x_axis_label="X-axis name", y_axis_label=stat_wgt, 
        tools="reset,box_zoom,pan,wheel_zoom,save", background_fill_color="#fafafa",
        plot_width=800, plot_height=500
    )

    # Plot the histogram squares...
    bkp.quad(
        top=df_for_plt[stat_wgt], bottom=0,
        left=df_for_plt['x_left'], right=df_for_plt['x_right'],
        fill_color="khaki", line_color="white", legend_label="Weight"
    )
    # ...at the bottom of the graph
    bkp.y_range = bokeh.models.ranges.Range1d(
        **expand_lims(df_for_plt[stat_wgt], 0, 1.2, 0)
    )

    bkp.legend.location = "top_left"
    bkp.legend.click_policy="hide"

    # Plot the weight average statistic points joined by straight lines
    # Set up the secondary axis
    bkp.extra_y_ranges['y_range_2'] = bokeh.models.ranges.Range1d(
        **expand_lims(df_for_plt[[stat_var + '_wgt_av' for stat_var in stat_vars]])
    )
    bkp.add_layout(bokeh.models.axes.LinearAxis(
        y_range_name='y_range_2',
        axis_label="Weighted average statistic"
    ), 'right')

    for var_num, stat_var in enumerate(stat_vars):
        # The following parameters need to be passed to both circle() and line()
        stat_line_args = {
            'x': df_for_plt['x_point'],
            'y': df_for_plt[stat_var + '_wgt_av'],
            'y_range_name': 'y_range_2',
            'color': cols[var_num],
            'legend_label': stat_var,
        }
        bkp.circle(**stat_line_args, size=4)
        bkp.line(**stat_line_args)
    
    return(bkp)


#########################
# Running interactively #
#########################
if __name__ == '__main__':
    print("bucketplot source script has completed")
