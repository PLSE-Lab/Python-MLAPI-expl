"""Functions to interactively cut the data into buckets and plot the results"""

# Useful to have a version number for checking updates
__version__ = '0.0.2'

############
# Contents #
############
# - Setup
# - get_cut_grps()
# - get_agg_plot_data()
# - create_plot()

#########
# Setup #
#########
# Import external modules
import numpy as np
import pandas as pd

from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.models.ranges import Range1d
from bokeh.models.axes import LinearAxis

################
# get_cut_grps #
################
def get_cut_grps(df, cut_by, n_bins):
    """
    df, DataFrame: to be cut into buckets
    cut_by, str: column name to use to make the cut
    n_bins, int or string: number of bins, or specific configuration
    
    Return: Object that can be used in groupby()
    """
    if isinstance(n_bins, int):
        return(pd.cut(df[cut_by], bins=n_bins))
    if n_bins == 'cat':
        return(df[cut_by])
    if n_bins == 'all':
        bins = np.sort(df[cut_by].unique())
        offset = np.min(np.diff(bins)) / 2
        bins = np.insert(bins, 0, 2*bins[0] - bins[1]) + offset
        return(pd.cut(df[cut_by], bins=bins))

#####################
# get_agg_plot_data #
#####################
def get_agg_plot_data(
    data_df,
    stat_cols=None, stat_wgt=None,
    cut_by=None, n_bins=None, order_by=None, bucket_wgt=None,
    x_axis_var=None,
    set_config=None,
):
    """
    Cut into buckets, and calculate the resulting stats
    
    stat_cols: List of column names for the weighted average to be calculated. Default none.
    stat_wgt: Weight for the purpose of weighted average statistics. Must be the same for all stat_cols.
    
    cut_by: Column name to cut by or 'cum_wgt' if cutting by 'bucket_wgt'
    n_bins: Number of bins to attempt to cut into, or a specific string
    If cutting into weighted quantiles:
        cut_by: Set this to 'cum_wgt'
        bucket_wgt: Column name of the weights
        order_by: Column name of the quantile field
    
    x_axis_var: Column name of variable to give on the x-axis
    
    set_config: Pre-defined configurations, so you don't need to set the other variables:
        'lift': Split into buckets for calculating lift. 
        
    Return: Grouped and aggregated DataFrame
    """
    # Set defaults
    if stat_cols is None:
        stat_cols=[]
    if isinstance(stat_cols, str):
        stat_cols = [stat_cols]
    if set_config == "lift":
        if order_by is None:
            order_by = stat_cols[0]
        if cut_by is None:
            cut_by = 'cum_wgt'
        if x_axis_var is None:
            x_axis_var = cut_by
        if n_bins is None:
            n_bins = 10
    if cut_by is None:
        cut_by = order_by
    if x_axis_var is None:
        x_axis_var = cut_by
    if n_bins is None:
        n_bins=30

    plt_data_df = data_df\
    .assign(
        wgt=lambda x: 1 if bucket_wgt is None else x[bucket_wgt],
        stat_wgt=lambda x: 1 if stat_wgt is None else x[stat_wgt],
        **{
            stat_col + "_wgt_sum": 
            lambda x, stat_col=stat_col: x[stat_col] * x['stat_wgt']
            for stat_col in stat_cols
        },
    ).rename_axis(index='index').sort_values([order_by, 'index']).assign(
        cum_wgt_raw=lambda x: x.wgt.cumsum(),
        cum_wgt=lambda x: x.groupby(order_by).cum_wgt_raw.transform('max'),
        grp=lambda df: get_cut_grps(df, cut_by, n_bins)
    ).groupby('grp', sort=False).agg(
        n_obs=('grp', 'size'), 
        wgt_sum=('wgt', 'sum'),
        stat_wgt_sum=('stat_wgt', 'sum'),
        **{stat_col + "_wgt_sum": (stat_col + "_wgt_sum", 'sum') for stat_col in stat_cols},
        x_min=(x_axis_var, 'min'),
        x_max=(x_axis_var, 'max'),
    ).pipe(lambda x: (
        x.reset_index(drop=True).pipe(lambda x: (
            x.set_index(pd.interval_range(start=-0.5, periods=x.shape[0], freq=1.))
        )) if n_bins == 'cat' 
        else x.set_index(x.index.categories)
    )).assign(
        **{
            stat_col + "_wgt_av": 
            lambda x, stat_col=stat_col: x[stat_col + "_wgt_sum"] / x.stat_wgt_sum
            for stat_col in stat_cols
        },
        x_left=lambda x: np.where(x.x_min == x.x_max , x.index.left, x.x_min),
        x_right=lambda x: np.where(x.x_min == x.x_max, x.index.right, x.x_max),
        x_mid=lambda x: (x.x_right + x.x_left) / 2,
    )
    return(plt_data_df)

###############
# create_plot #
###############
def create_plot(
    plt_data_df,
    stat_cols,
    n_bins=None
):
    """
    Given the grouped and aggregated DataFrame, get the plot object
    
    plt_data_df: DataFrame in the form of the result of get_agg_plot_data()
    stat_cols, n_bins: Same as the arguments to get_agg_plot_data()
    """
    # Set defaults
    if stat_cols is None:
        stat_cols=[]
    if isinstance(stat_cols, str):
        stat_cols = [stat_cols]
    
    bkplt = figure(
        title="Predicted vs Actual chart", x_axis_label='Pred val', y_axis_label="Exposure", 
        tools="reset,box_zoom,pan,wheel_zoom,save", background_fill_color="#fafafa",
        plot_width=800, plot_height=500
    )
    bkplt.quad(
        top=plt_data_df.wgt_sum, bottom=0, left=plt_data_df.x_left, right=plt_data_df.x_right,
        fill_color="khaki", line_color="white", legend_label="Exposure"
    )
    bkplt.y_range=Range1d(0, plt_data_df.wgt_sum.max() / 0.5)

    stats_range = {
        'min': plt_data_df[[stat_col + "_wgt_av" for stat_col in stat_cols]].min().min(),
        'max': plt_data_df[[stat_col + "_wgt_av" for stat_col in stat_cols]].max().max(),
    }
    pct_y_buffer = {
        'top': 0.1, 'bottom': 0.05
    }
    y_range2_name = 'y_range2_name'
    bkplt.extra_y_ranges[y_range2_name] = Range1d(
        stats_range['min'] - (stats_range['max'] - stats_range['min']) * pct_y_buffer['bottom'], 
        stats_range['max'] + (stats_range['max'] - stats_range['min']) * pct_y_buffer['top'],
    )
    ax_new = LinearAxis(y_range_name=y_range2_name, axis_label="Average response")
    bkplt.add_layout(ax_new, 'right')

    
    for col_name, color in zip(stat_cols, ['purple', 'green']):
        bkplt.circle(
            plt_data_df.x_mid, plt_data_df[col_name + "_wgt_av"], 
            color=color, size=4,
            y_range_name=y_range2_name,
            legend_label=col_name
        )
        bkplt.line(
            plt_data_df.x_mid, plt_data_df[col_name + "_wgt_av"], 
            color=color,
            y_range_name=y_range2_name,
            legend_label=col_name
        )

    bkplt.grid.grid_line_color = "white"
    bkplt.legend.location = "top_left"
    bkplt.legend.click_policy="hide"

    if n_bins == "cat":
        x_tick_labs = plt_data_df.x_min.astype(str).reset_index(drop=True).to_dict()
        bkplt.xaxis.ticker = list(x_tick_labs.keys())
        bkplt.xaxis.major_label_overrides = x_tick_labs
    
    return(bkplt)

if __name__ == '__main__':
    print("bucketplot source script has completed")
