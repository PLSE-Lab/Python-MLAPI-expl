from datetime import datetime
import pandas as pd

eps = 1e-6

str_bools = ['on', 'off', 'true', 'false', '+', '-']
int_bools = [1, 0]
true_bools = ['on', 'true', '+', 1]

errors_possible_values = ['raise', 'filter_out', 'coerce', 'ignore']


def is_datetime(value):
    """
    Checks whether it is a datetime value (it is not equal NaT and None).
    :param value: value for checking
    :return: true, if it is a datetime
             false, otherwise
    """
    return value == value and value is not None


def convert_to_pretty_df(df_or_serie, errors, columns=None, column_type=None, column_datetime_params=None):
    """
    Convert to pandas.DataFrame or pandas.Series with appropriate column type.
    For example, convert column with 'on'/'off' values to boolean column.
    :param df_or_serie: pandas.DataFrame or pandas.Series for converting.
    :param errors: should be one of following values:
        1) 'raise' - raise an exception if exist value that does not fit in required type,
        2) 'filter_out' - filter out rows where at least one element does not fit in required type,
        3) 'coerce' - set None, if value does not fit in required type,
        4) 'skip' - leave as is, if value does not fit in required type.
    :param columns: list of names of columns that will be taken into account for input DataFrame.
        if columns is None then all columns will be considered.
        If df_or_serie is a pandas.Series then value is ignored.
    :param column_type: map of column -> appropriate column type.
        If column_type does not have type for any column, then it will be chosen automatically.
        If df_or_serie is a pandas.Series then value is ignored.
    :param column_datetime_params: map of column -> datetime params.
        Datetime params will be used in pandas.to_datetime method.
    :return: raise an exception if there are errors in input arguments,
        returns pandas.DataFrame or pandas.Series with appropriate column type otherwise.
    """
    if isinstance(df_or_serie, pd.DataFrame) or isinstance(df_or_serie, pd.Series):
        if errors not in errors_possible_values:
            raise ValueError("errors should be one of following values: raise, filter_out, coerce or ignore")
        if columns is not None and not isinstance(columns, list):
            raise ValueError("columns should be list or None")
        if column_type is not None and not isinstance(column_type, dict):
            raise ValueError("column_type should be dict or None")
        if column_datetime_params is None:
            pass
        elif isinstance(column_datetime_params, dict):
            for key, value in column_datetime_params.items():
                if not isinstance(value, dict):
                    raise ValueError("value in column_datetime_params should be dict (key is " + key + ")")
        else:
            raise ValueError("column_datetime_params should be dict or None")
        if isinstance(df_or_serie, pd.DataFrame):
            return do_convert_to_pretty_df(df_or_serie, errors, columns, column_type, column_datetime_params)
        else:
            array, failed_indices, type = do_convert_to_pretty_serie(df_or_serie, errors,
                                                                     column_type, column_datetime_params)
            result = pd.Series(data=array)
            result.drop(df.index[list(failed_indices)], inplace=True)
            return result.astype(type)
    else:
        raise ValueError("df_or_serie should be pandas.DataFrame or pandas.Series")


def do_convert_to_pretty_df(df, errors, columns, column_type, column_datetime_params):
    """
    Internal method for converting to pandas.DataFrame with appropriate column types.
    :param df: pandas.DataFrame for converting
    :param errors: should be one of following values:
        1) 'raise' - raise an exception if exist value that does not fit in required type,
        2) 'filter_out' - filter out rows where at least one element does not fit in required type,
        3) 'coerce' - set None, if value does not fit in required type,
        4) 'skip' - leave as is, if value does not fit in required type.
    :param columns: list of columns that will be taken into account for input DataFrame.
        if columns is None then all columns will be considered.
    :param column_type: map of column -> appropriate column type.
        If column_type does not have type for any column, then it will be chosen automatically.
    :param column_datetime_params: map of column -> datetime params.
        Datetime params will be used in pandas.to_datetime method.
    :return: pandas.DataFrame with appropriate column type otherwise.
    """
    result = pd.DataFrame()
    indices_to_remove = set()
    types = dict()
    for column in df.columns:
        serie = df[column]
        if columns is not None:
            if column in columns:
                current_column_type = None if column_type is None else column_type.get(column)
                datetime_params = None if column_datetime_params is None else column_datetime_params.get(column)
                serie, failed_indices, appropriate_type = do_convert_to_pretty_serie(serie, errors,
                                                                                     current_column_type,
                                                                                     datetime_params)
                types[column] = appropriate_type
                indices_to_remove = indices_to_remove.union(failed_indices)
            else:
                pass
        else:
            current_column_type = None if column_type is None else column_type.get(column)
            datetime_params = None if column_datetime_params is None else column_datetime_params.get(column)
            serie, failed_indices, appropriate_type = do_convert_to_pretty_serie(serie, errors,
                                                                     current_column_type, datetime_params)
            types[column] = appropriate_type
            indices_to_remove = indices_to_remove.union(failed_indices)
        result[column] = serie
    result.drop(df.index[list(indices_to_remove)], inplace=True)
    if errors == 'filter_out':
        for column, appropriate_column_type in types.items():
            result[column] = result[column].astype(appropriate_column_type)
    return result


def do_convert_to_pretty_serie(serie, errors, column_type, datetime_params):
    """
    Internal method for converting to pandas.Series with appropriate column types.
    :param serie: pandas.Series for converting
    :param errors: should be one of following values:
        1) 'raise' - raise an exception if exist value that does not fit in required type,
        2) 'filter_out' - filter out rows where at least one element does not fit in required type,
        3) 'coerce' - set None, if value does not fit in required type,
        4) 'skip' - leave as is, if value does not fit in required type.
    :param column_type: appropriate column type.
        If column_type is None, then type of serie will be chosen automatically.
    :param datetime_params: Datetime params will be used in pandas.to_datetime method.
    :return: (pandas.Series, array of failed indices, appropriate type).
    """
    if column_type is None:
        column_type = calculate_best_type(serie, datetime_params)
    if column_type == bool:
        array, failed_indices = to_boolean(serie, errors)
        return array, failed_indices, 'bool'
    elif column_type == int:
        array, failed_indices = to_int(serie, errors)
        return array, failed_indices, 'int64'
    elif column_type == float:
        array, failed_indices = to_float(serie, errors)
        return array, failed_indices, 'float64'
    elif column_type == datetime:
        array, failed_indices = to_datetime(serie, errors, datetime_params)
        return array, failed_indices, 'datetime64[ns]'
    elif column_type == str:
        array, failed_indices = to_str(serie, errors)
        # Pandas uses the object dtype for storing strings.
        # https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#dtypes
        return array, failed_indices, 'str'
    else:
        raise ValueError("Unknown column type " + column_type)


def to_boolean(serie, errors):
    """
    Convert pandas.Series to pandas.Series with booleans
    :param serie: pandas.Series for converting
    :param errors: should be one of following values:
        1) 'raise' - raise an exception if exist value that does not fit in required type,
        2) 'filter_out' - filter out rows where at least one element does not fit in required type,
        3) 'coerce' - set None, if value does not fit in required type,
        4) 'skip' - leave as is, if value does not fit in required type.
    :return: pandas.Series with booleans
    """

    def do_on_error():
        """
        Closure changes failed_indices and result variables depending on errors variable.
        :return: raises an error, or changes failed_indices and result variables
        """
        if errors == 'raise':
            raise ValueError("Value is not like a boolean in row " + str(idx))
        elif errors == 'filter_out':
            failed_indices.append(idx)
            result.append(element)
        elif errors == 'coerce':
            result.append(None)
        elif errors == 'ignore':
            result.append(element)

    result = []
    failed_indices = []
    for idx, element in enumerate(serie):
        if isinstance(element, bool):
            result.append(element)
        elif isinstance(element, str):
            if element.lower() in str_bools:
                if element.lower() in true_bools:
                    result.append(True)
                else:
                    result.append(False)
            else:
                do_on_error()
        elif isinstance(element, int):
            if element in [0, 1]:
                if element == 1:
                    result.append(True)
                else:
                    result.append(False)
            else:
                do_on_error()
        else:
            do_on_error()
    return result, failed_indices


def to_datetime(serie, errors, datetime_params):
    """
    Convert pandas.Series to pandas.Series with datetimes
    :param serie: pandas.Series for converting
    :param errors: should be one of following values:
        1) 'raise' - raise an exception if exist value that does not fit in required type,
        2) 'filter_out' - filter out rows where at least one element does not fit in required type,
        3) 'coerce' - set None, if value does not fit in required type,
        4) 'skip' - leave as is, if value does not fit in required type.
    :param datetime_params: datetime_params to configure parsing to datetime
    :return: pandas.Series with datetimes
    """
    result = []
    failed_indices = []
    for idx, element in enumerate(serie):
        value = convert_to_datetime_with_params(element, datetime_params)
        if is_datetime(value):
            result.append(value)
        else:
            if errors == 'raise':
                raise ValueError("Value is not like a datetime in row " + str(idx))
            elif errors == 'filter_out':
                failed_indices.append(idx)
                result.append(element)
            elif errors == 'coerce':
                result.append(None)
            elif errors == 'ignore':
                result.append(element)
    return result, failed_indices


def to_str(serie, errors):
    """
        Convert pandas.Series to pandas.Series with strings
        :param serie: pandas.Series for converting
        :param errors: should be one of following values:
            1) 'raise' - raise an exception if exist value that does not fit in required type,
            2) 'filter_out' - filter out rows where at least one element does not fit in required type,
            3) 'coerce' - set None, if value does not fit in required type,
            4) 'skip' - leave as is, if value does not fit in required type.
        :return: pandas.Series with strings
        """
    result = []
    failed_indices = []
    for idx, element in enumerate(serie):
        if isinstance(element, str):
            result.append(element)
        else:
            if errors == 'raise':
                raise ValueError("Value is not like a str in row " + str(idx))
            elif errors == 'filter_out':
                failed_indices.append(idx)
                result.append(element)
            elif errors == 'coerce':
                result.append(None)
            elif errors == 'ignore':
                result.append(element)
    return result, failed_indices


def to_int(serie, errors):
    """
    Convert pandas.Series to pandas.Series with ints
    :param serie: pandas.Series for converting
    :param errors: should be one of following values:
        1) 'raise' - raise an exception if exist value that does not fit in required type,
        2) 'filter_out' - filter out rows where at least one element does not fit in required type,
        3) 'coerce' - set None, if value does not fit in required type,
        4) 'skip' - leave as is, if value does not fit in required type.
    :return: pandas.Series with ints
    """
    result = []
    failed_indices = []
    for idx, element in enumerate(serie):
        try:
            value = int(element)
            result.append(value)
        except (TypeError, ValueError):
            try:
                value = float(element)
                int_val = int(value)
                if abs(value - int_val) < eps:
                    result.append(int_val)
                else:
                    raise ValueError("")
            except (TypeError, ValueError):
                if errors == 'raise':
                    raise ValueError("Value is not like a int in row " + str(idx))
                elif errors == 'filter_out':
                    failed_indices.append(idx)
                    result.append(element)
                elif errors == 'coerce':
                    result.append(None)
                elif errors == 'ignore':
                    result.append(element)
    return result, failed_indices


def to_float(serie, errors):
    """
    Convert pandas.Series to pandas.Series with floats
    :param serie: pandas.Series for converting
    :param errors: should be one of following values:
        1) 'raise' - raise an exception if exist value that does not fit in required type,
        2) 'filter_out' - filter out rows where at least one element does not fit in required type,
        3) 'coerce' - set None, if value does not fit in required type,
        4) 'skip' - leave as is, if value does not fit in required type.
    :return: pandas.Series with floats
    """
    result = []
    failed_indices = []
    for idx, element in enumerate(serie):
        try:
            value = float(element)
            result.append(value)
        except (TypeError, ValueError):
            if errors == 'raise':
                raise ValueError("Value is not like a float in row " + str(idx))
            elif errors == 'filter_out':
                failed_indices.append(idx)
                result.append(element)
            elif errors == 'coerce':
                result.append(None)
            elif errors == 'ignore':
                result.append(element)
    return result, failed_indices


def calculate_best_type(serie, datetime_params):
    """
    Calculates the best type for given serie.
    :param serie: serie for calculation.
    :param datetime_params:  Datetime params will be used in pandas.to_datetime method.
    :return: the best type for given serie.
    """
    unique_values = serie.unique()
    bool_count = sum(isinstance(x, str) and x.lower() in str_bools or isinstance(x, bool)
                     or x in [0, 1] for x in unique_values)
    datetime_count = sum(is_datetime(convert_to_datetime_with_params(x, datetime_params)) for x in unique_values)
    str_count = sum(isinstance(x, str) for x in unique_values)
    int_count = 0
    for element in unique_values:
        try:
            int(element)
            int_count += 1
        except (TypeError, ValueError):
            try:
                value = float(element)
                int_val = int(value)
                if abs(value - int_val) < eps:
                    int_count += 1
                else:
                    raise ValueError("")
            except (TypeError, ValueError):
                pass
    float_count = 0
    for element in unique_values:
        try:
            float(element)
            float_count += 1
        except (TypeError, ValueError):
            pass
    types = [(bool, bool_count), (int, int_count), (float, float_count), (datetime, datetime_count), (str, str_count)]
    probably_type = max(types, key=lambda x: x[1])[0]
    if probably_type == str and datetime_count > int_count:
        return datetime
    else:
        return probably_type


def convert_to_datetime_with_params(x, datetime_params):
    """
    Convert x to datetime using pandas.to_datetime and datetime_params as pandas.to_datetime arguments.
    :param x: value for conversion.
    :param datetime_params: pandas.to_datetime arguments.
    :return: datetime if x is datetime,
        None otherwise.
    """
    if datetime_params is None:
        datetime_params = {'errors': 'coerce'}
    else:
        datetime_params['errors'] = 'coerce'
    return pd.to_datetime(x, **datetime_params)
