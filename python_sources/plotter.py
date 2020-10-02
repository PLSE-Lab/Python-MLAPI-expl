import pandas as pd
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import DatetimeTickFormatter
from bokeh.models import Range1d


WEEK_N_PALETTE = {1: "#581845", 2: "#900C3F", 3: "#C70039", 4: "#FF5733", 5: "#FFC300"}

def wide_tools():
    return "pan,ywheel_zoom,xwheel_zoom,reset,save"

def get_time_series_figure(width=400, height=400, title=""):
    fig = figure(plot_width=width, plot_height=height, x_axis_type='datetime', tools=wide_tools(), title=title)
    fig.xaxis.formatter = DatetimeTickFormatter(
        hours=["%H:%M %d/%b"],
        days=["%d/%b"],
        months=["%d/%b/%Y"],
        years=["%d/%b/%Y"],
    )
    return fig

def plot_time_series_count(str_datetimes, values, color, title="", relative_y_axis=False, alpha=0.8, width=900,
                           height=300, line_width=1, legend=None, p=None):
    if not p:
        p = get_time_series_figure(width=width, height=height, title=title)

    datetimes = pd.to_datetime(str_datetimes)
    p.circle(datetimes, values, size=3, color=color, alpha=alpha, legend=legend)
    p.line(datetimes, values, line_width=line_width, color=color, alpha=alpha)

    if not relative_y_axis:
        p.y_range = Range1d(0, max(values) * 1.1)

    return p

def time_series_count_painted_holidays(data, p, color="cyan", alpha=0.9):
    holidays = data[data["IsHoliday"] == True]
    holidays["Date"] = holidays["Date"].apply(pd.to_datetime)
    grouped_sales = holidays.groupby("Date")["Weekly_Sales"].median().to_frame()
    p.diamond("Date", "Weekly_Sales", line_color=color, size=12, line_width=2,
              source=ColumnDataSource(grouped_sales), alpha=alpha, fill_color=None)
    return p

def time_series_count_painted(data, palette=WEEK_N_PALETTE, title="", width=900, height=300, alpha=0.9):
    p = get_time_series_figure(width=width, height=height, title=title)

    grouped_sales = data.groupby("Date")["Weekly_Sales"].median().to_frame()
    data["week_color"] = data["week_n"].apply(lambda week_n: palette[week_n])
    week_n_colors = data[["Date", "week_color"]].drop_duplicates().set_index("Date")
    week_ns = data[["Date", "week_n"]].drop_duplicates().set_index("Date")

    plot_data = grouped_sales.merge(week_n_colors, how="left", left_index=True, right_index=True)
    plot_data = plot_data.merge(week_ns, how="left", left_index=True, right_index=True).reset_index()

    plot_data["Date"] = plot_data["Date"].apply(pd.to_datetime)
    source = ColumnDataSource(plot_data)

    p.circle("Date", "Weekly_Sales", size=4, color="week_color", alpha=alpha, legend="week_n", source=source)
    p.line("Date", "Weekly_Sales", line_width=1, color="gray", alpha=0.3, source=source)

    p.legend.title = 'week_n sales'

    return p


def plot_error_values(df, group_by_col, values_col, title="", drop_quantile=0.20,
                      width=400, height=300):
    group = df[df[values_col].notna()].groupby(group_by_col)

    # some pseudo data
    xs = group.apply(lambda g : g.name).astype(str)
    yerrs_pos = group[values_col].apply(lambda sd : sd.quantile(1 - drop_quantile))
    yerrs_neg = group[values_col].apply(lambda sd : sd.quantile(drop_quantile))
    ys = group[values_col].median()

    # plot the points
    p = figure(x_range=xs, title=title, width=width, height=height, tools="pan,ywheel_zoom,reset,save")
    p.xaxis.axis_label = group_by_col
    p.yaxis.axis_label = values_col

    p.circle(xs, ys, color='navy', size=10, line_alpha=0)
    p.circle(xs, ys, color='magenta', size=5, line_alpha=0)

    # create the coordinates for the errorbars
    err_xs = []
    err_ys = []

    for x, y, yerr_pos, yerr_neg in zip(xs, ys, yerrs_pos, yerrs_neg):
        err_xs.append((x, x))
        err_ys.append((yerr_neg, yerr_pos))

    # plot them
    p.multi_line(err_xs, err_ys, color='magenta', line_width=2)

    return p
