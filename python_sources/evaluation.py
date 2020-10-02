import pandas as pd


def build_submission_df(test_df, target_predicted, store_colname="Store", dept_colname="Dept", date_colname="Date"):
    if len(test_df) != len(target_predicted):
        raise Exception(
            "Teste dataset and target columns must have the same lengths: {}, {}".format(len(test_df),
                                                                                         len(target_predicted)))
    submission_df = pd.DataFrame()
    submission_df["Id"] = test_df[store_colname].astype(str) + "_" + \
                          test_df[dept_colname].astype(str) + "_" +\
                          test_df[date_colname].astype(str)
    submission_df["Weekly_Sales"] = target_predicted
    return submission_df.set_index("Id")

def evaluate(submission_df, validation, valid_store_colname="Store", valid_dept_colname="Dept",
             valid_date_colname="Date", valid_weekly_sales="Weekly_Sales"):

    validation = validation[[valid_store_colname, valid_dept_colname, valid_date_colname,
                             valid_weekly_sales, "IsHoliday"]].copy()
    validation["Id"] = validation[valid_store_colname].astype(str) + "_" + \
                       validation[valid_dept_colname].astype(str) + "_" + \
                       validation[valid_date_colname].astype(str)
    validation = validation.set_index("Id")
    abs_diff = (validation[valid_weekly_sales] - submission_df[valid_weekly_sales]).apply(abs)
    w = validation["IsHoliday"].replace({True: 5, False: 1})
    w = w.reindex(validation.index)

    return (w * abs_diff).sum() / w.sum()

def dummy_baseline_data(data):
    dummy_baseline = []

    for (store_dept, wm_date), g in data.groupby(["store_dept", "wm_date"]):
        sorted_group = g[["year", "Weekly_Sales", "Size", "Store", "Dept", "Date", "IsHoliday"]].sort_values("year")
        store = sorted_group["Store"].iloc[0]
        dept = sorted_group["Dept"].iloc[0]
        date = sorted_group["Date"].iloc[0]
        is_holiday = sorted_group["IsHoliday"].iloc[0]
        
        dummy_baseline_row = {"store_dept": store_dept, "wm_date": wm_date,
                              "Store": store, "Dept": dept, "Date": date, "IsHoliday": is_holiday}
        
        for year_i in range(len(sorted_group)):
            year_n_label = "year" + str(year_i)
            year_sales_label = year_n_label + "_sales"
            size_n_label = year_n_label + "_size"

            year_value = sorted_group.iloc[year_i]["year"]
            x_value = sorted_group.iloc[year_i]["Weekly_Sales"]
            size_value = sorted_group.iloc[year_i]["Size"]

            dummy_baseline_row[year_n_label] = year_value
            dummy_baseline_row[year_sales_label] = x_value
            dummy_baseline_row[size_n_label] = size_value

        dummy_baseline.append(dummy_baseline_row)

    return pd.DataFrame(dummy_baseline)