
import datetime

import numpy as np
import pandas as pd


HOLIDAYS = [
    "2020-03-08",
    "2020-03-25",
    "2020-03-30",
    "2020-03-31"
]

SLA = pd.DataFrame(
    data=[
        ["manila","manila",3],
        ["manila","luzon",5],
        ["manila","visayas",7],
        ["manila","mindanao",7],
        ["luzon","manila",5],
        ["luzon","luzon",5],
        ["luzon","visayas",7],
        ["luzon","mindanao",7],
        ["visayas","manila",7],
        ["visayas","luzon",7],
        ["visayas","visayas",7],
        ["visayas","mindanao",7],
        ["mindanao","manila",7],
        ["mindanao","luzon",7],
        ["mindanao","visayas",7],
        ["mindanao","mindanao",7]
    ],
    columns=["origin", "destination", "sla"]
)


def parse_data(raw: pd.DataFrame, sla: pd.DataFrame) -> pd.DataFrame:
    """
    Load data and parse addresses

    Parameters
    ----------
    raw : pd.DataFrame
        raw dataframe, expected columns are
        - orderid
        - pick
        - 1st_deliver_attempt
        - 2nd_deliver_attempt
        - buyeraddress
        - selleraddress
    sla : pd.DataFrame
        sla conditions

    Returns
    -------
    pd.DataFrame
        parsed data, expected columns are
        - orderid
        - is_late
    """

    ret = raw.copy()

    # address

    ret["buyeraddress"] = ret["buyeraddress"].str.split(' ').str[-1].str.lower()
    ret["selleraddress"] = ret["selleraddress"].str.split(' ').str[-1].str.lower()

    # date

    for col in ["pick", "1st_deliver_attempt", "2nd_deliver_attempt"]:
        times = pd.to_datetime(ret[col],unit='s') + pd.to_timedelta(8, unit='h')
        ret[col] = times.dt.date

    # merge sla

    ret = pd.merge(
        ret,
        sla,
        left_on=["buyeraddress", "selleraddress"],
        right_on=["destination", "origin"]
    )

    ret = ret.drop(columns=["buyeraddress", "selleraddress", "destination", "origin"])

    # get date deltas

    ret["delta_1"] = delta_working_days(
        ret["pick"],
        ret["1st_deliver_attempt"]
    )

    ret["delta_2"] = delta_working_days(
        ret["1st_deliver_attempt"],
        ret["2nd_deliver_attempt"]
    )

    # get label

    ret["on_time"] = (ret["delta_1"] <= ret["sla"]) & \
        ((ret["delta_2"] <= 3) | (ret["delta_2"].isnull()))

    ret["is_late"] = ~ret["on_time"]

    return ret


def delta_working_days(
    dates_1: pd.Series,
    dates_2: pd.Series
) -> np.array:
    """
    Get the number of working days between two series of dates

    Parameters
    ----------
    dates_1 : pd.Series
        beginning dates, can contain pd.NaT
    dates_2 : pd.Series
        ending dates, can contain pd.NaT

    Returns
    -------
    np.array
        deltas working days,
        np.nan when beginning or ending date is unspecified
    """

    mask_non_null = dates_1.notna() & dates_2.notna()

    deltas = np.busday_count(
        dates_1.loc[mask_non_null],
        dates_2.loc[mask_non_null],
        weekmask="1111110", # sunday is not working day
        holidays=HOLIDAYS
    )

    ret = np.empty(len(dates_1))

    ret[:] = np.nan
    ret[mask_non_null] = deltas

    return ret


def save_labels(
    df: pd.DataFrame,
    path: str = "submission.csv"
):
    """
    Save submission file

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with 'orderid' and 'is_late' columns
    path : str, optional
        output path, by default "submission.csv"
    """

    ret = df.copy()
    ret["is_late"] = ret["is_late"].astype("int16")

    print(f"{len(ret)} entries")

    ret[["orderid", "is_late"]].to_csv(path, index=False)

    
def main():
    """
    Execute the main routine to output predictions file
    from input files
    """
    
    path_to_data = "/kaggle/input/open-shopee-code-league-logistic/delivery_orders_march.csv"
    date_suffix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = f"/kaggle/working/submission_{date_suffix}.csv"
    
    
    raw = pd.read_csv(path_to_data)

    orders = parse_data(raw, SLA)

    save_labels(orders, output_path)

main()