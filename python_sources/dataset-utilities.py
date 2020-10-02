import pandas as pd


def categorify(df, threshold=0.1):
    """Change columns to type ``category`` if they have <= ``threshold`` unique values.
    This is useful to save memory in the cases where you have a large amount of data, and 
    some columns have a small number of options (categories).
    :param df: pandas DataFrame
    :param threshold: float [0, 1]. Percentage of unique values required to conver the 
        column to a categorical dtype
    """
    n = len(df)
    obj_cols = df.select_dtypes('object').columns
    lt_thresh_cols = df.columns[df.nunique().lt(n * threshold)] 
    category_cols = obj_cols.intersection(lt_thresh_cols)

    dtypes = dict(zip(category_cols, ['category' for i in range(len(category_cols))]))
    df = df.astype(dtypes)
    return df


def value_counts_plus(series,
                      dropna=False,
                      show_top=10,
                      sort_others=False,
                      style=True,
                      background_gradient='cividis'):
    """Provide a few ways of showing counts of values of items in ``series``.
    :param list series: pandas Series or list
    :param bool dropna: Whether or not to drop missing values
    :param int show_top: Display only this many rows from the result
    :param bool sort_others: Whether or not to place "Others" in the bottom (default) or
                             in its sorted order position
    :param bool style: Whether or not to style values for easier reading. If set 
                       to ``True`` the result would not be a DataFrame, and cannot 
                       be further manipulated. Set the value to ``False`` to get a
                       DataFrame as the return value. 
    """
    series = pd.Series(series)
    series.name = series.name or 'data'
    col = series.name or 'data'
    val_counts = series.value_counts(dropna=dropna)
    if len(val_counts) > show_top:
        val_counts = val_counts[:show_top].append(pd.Series(val_counts[show_top:].sum(),
                                                index=['Others:'], name=col))
        if sort_others:                                                
            val_counts = val_counts.sort_values(ascending=False)                                                
        show_top += 1
    count_df = (val_counts
                .to_frame()
                .assign(cum_count=lambda df: df[col].cumsum(),
                        perc=lambda df: df[col].div(df[col].sum()),
                        cum_perc=lambda df: df['perc'].cumsum())
                .reset_index()
                .rename(columns={'index': col, col: 'count'}))
    if not style:
        return count_df.head(show_top)
    return (count_df.
            head(show_top).style
            .format({'count': '{:,}', 'cumsum': '{:,}', 
                     'perc': '{:.1%}', 
                     'cum_perc': '{:.1%}'})
            .background_gradient(background_gradient))




def count_separated_fields(series, sep, dropna=False, show_top=10, style=True):
    """Split rows of a Series by ``sep``, extract item and count values."""
    split_series = pd.Series(series.str.cat(sep=sep).split(sep))
    return value_counts_plus(split_series, dropna=dropna, show_top=show_top, style=style)


def _highlight_found(series, regex, case):
    found = series.astype(str).str.contains(regex, case=case)
    return ['background-color: #8AA97C; font-weight: bold' if v else '' for v in found]


def df_search(df, regex, case=False, highlight=True):
    """Return all rows where any column contains ``regex``.
    :param df: pandas DataFrame
    :param regex: regular expression
    :param case: Whether or ot to take case into consideration
    :param highlight: Whether or not to highlight the found values (applies in
    Jupyter notebooks.
    """
    result_index = (df
                    .select_dtypes('object')
                    .apply(lambda series: series.astype(str).str.contains(regex, case=case))
                    .apply(lambda row: any(row), axis=1))
    return df[result_index].style.apply(_highlight_found, regex=regex, case=case) if highlight else df[result_index]
