#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This notebook offers class M5ACC_SubmissionGatekeeper
# which you can use to validate consistency of your
# M5 accuracy forecasts
#
# It prvides the following checks: 
#
# 1. No NaN-s and negative values in the forecast;
#
# 2. Forecasted DAILY volumes for every item 
# and their aggregations 
# should be within historical [min, 2*max] ranges 
# for daily volumes of these items 
# and their aggregations across all 12 levels;
# 
# 3. Forecasted MONTHLY (defined as "calculated as a total 
# for 28 days in a row") volumes for every item 
# and their aggregations
# should be within historical [min, 2*max] ranges 
# for monthly volumes for these items 
# and their aggregations across all 12 levels;
#
# 4. No sales should be forecasted for discontinued items 
# (defined as items with no past sales 
# during the benchmarking period);
# 
# 5. Percent of non-zero values (defined as above 0.5) 
# in a 28-day period should be 
# within historical [min-10%, max+10%] range.


# In[ ]:


from __future__ import annotations
import numpy as np
import pandas as pd 
import copy
import os
import logging
from collections import namedtuple
from typing import Optional, Callable, Any


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Here we load M5 datasets\n#\nINPUT_DATA_DIR="/kaggle/input/m5-forecasting-accuracy"\nOUTPUT_DATA_DIR="."\n\nm5_sales_history_dtypes={"d_"+str(i):\'int32\' for i in range(1, 1914)}\nm5_sales_history_dtypes.update({\'id\':\'category\'\n                             ,\'item_id\':\'category\'\n                             ,\'dept_id\':\'category\'\n                             ,\'cat_id\':\'category\'\n                             ,\'store_id\':\'category\'\n                             ,\'state_id\':\'category\'\n                            })\n\nM5_SALES_TRAIN_VALIDATION_DF=pd.read_csv(\n    os.path.join(\n        INPUT_DATA_DIR,"sales_train_validation.csv")\n    ,dtype=m5_sales_history_dtypes)\n\nM5_SAMPLE_ACC_SUBMISSION_DF=pd.read_csv(os.path.join(\n    INPUT_DATA_DIR,"sample_submission.csv"))')


# In[ ]:


# There are 2 large cells below with classes defined inside the cells. 
# Skip these cells for now and
# go to the bottom of the notebbok to see an example of
# how the submission validator works


# In[ ]:


class LoggableObject:
    """ Base class for types that are able to log messages.
    A wrapper for standard Python logging functionality,
    extended with ability to automatically
    append a logged message with information
    about an object that generated the message."""

    def __init__(self
            , logger_name: str 
            , reveal_loggers_identity: bool = True
            , new_handler = None
            , new_level:Optional[int] = None
            , new_formatter = None
            ):
        self.logger = logging.getLogger(logger_name)
        self.reveal_identity = reveal_loggers_identity
        self.update_logger(new_level, new_handler, new_formatter)

    def reveal_self_names(self) -> str:
        """ Find the name(s) of variable(s) that hold self, if possible.

        The function uses a naive approach,
        it does not always find all the names"""

        all_names = set()

        for fi in inspect.stack():
            local_vars = fi.frame.f_locals
            names = {name for name in local_vars if local_vars[name] is self}
            all_names |= names

        all_names = list(all_names)
        if "self" in all_names:
            all_names.remove("self")

        if len(all_names) == 0:
            return ""
        elif len(all_names) == 1:
            return all_names[0]
        else:
            return str(all_names).replace("'", "")

    def __str__(self) -> str:
        description = f"Logged messages are labeled '{self.logger.name}' and send via {str(self.logger.handlers)}. "
        return description

    def update_logger(self
            , new_level:int = logging.DEBUG
            , new_handler = logging.StreamHandler()
            , new_formatter = logging.Formatter(
                '%(asctime)s %(name)s %(levelname)s: %(message)s',
                datefmt="%I:%M:%S")
            ) -> LoggableObject:
        if new_level is not None:
            self.logger.setLevel(new_level)

        if new_handler is not None:
            if new_level is not None:
                new_handler.setLevel(new_level)
            if new_formatter is not None:
                new_handler.setFormatter(new_formatter)

            for h in self.logger.handlers:
                if str(h) == str(new_handler):
                    self.logger.removeHandler(h)
                    self.logger.addHandler(new_handler)
                    break
            else:
                self.logger.addHandler(new_handler)

        return self

    def append_str_with_identity_info(self, a_str:str) -> str:
        if self.reveal_identity:
            self_id_str = self.reveal_self_names()
            self_id_str += ':' + type(self).__qualname__
            self_id_str = self_id_str.lstrip(':')
            full_len = len(a_str)
            a_str = a_str.rstrip('\n')
            num_eols = full_len - len(a_str)
            a_str += ' /* logged by ' + self_id_str + ' */'
            a_str += num_eols * '\n'
        return a_str

    def debug(self, msg:Optional[str] = None, *args, **kwargs):
        self.log(level=logging.DEBUG, msg=msg, *args, **kwargs)

    def info(self, msg:Optional[str] = None, *args, **kwargs):
        self.log(level=logging.INFO, msg=msg, *args, **kwargs)

    def warning(self, msg:Optional[str] = None, *args, **kwargs):
        self.log(level=logging.WARNING, msg=msg, *args, **kwargs)

    def error(self, msg:Optional[str] = None, *args, **kwargs):
        self.log(level=logging.ERROR, msg=msg, *args, **kwargs)

    def critical(self, msg:Optional[str] = None, *args, **kwargs):
        self.log(level=logging.CRITICAL, msg=msg, *args, **kwargs)

    def fatal(self, msg:Optional[str] = None, *args, **kwargs):
        self.log(level=logging.FATAL, msg=msg, *args, **kwargs)

    def log(self, level:int, msg:Optional[str] = None, *args, **kwargs):
        if msg is None:
            msg = str(self)
        msg = self.append_str_with_identity_info(msg)
        self.logger.log(level, msg, *args, **kwargs)  


# In[ ]:


class M5ACC_SubmissionGatekeeper(LoggableObject):
    """ A class that validates consistency of M5 accuracy forcasts"""

    def __init__(
            self
            ,traning_data_df
            ,sample_submission_df
            ,n_years_to_benchmark_against=1
            ,drop_Xmas_days=True
            ,n_df_lines_to_print_in_report=5
            ,logging_level = logging.DEBUG):
        
        # Let's perform basic consistency checks to make sure
        # baseline datasets traning_data_df and sample_submission_df
        # were not accidentally altered
        # after they were loaded from input .csv files

        self.level_labels_set = {"store_id", "item_id", "cat_id"
            , "state_id", "dept_id"}

        assert isinstance(traning_data_df, type(pd.DataFrame()))
        assert isinstance(sample_submission_df, type(pd.DataFrame()))
        assert sample_submission_df.shape == (60980, 29)
        assert len(traning_data_df) == 30490
        assert self.level_labels_set.issubset(traning_data_df.columns)

        assert set(traning_data_df.state_id.unique()
                   ) == {'CA', 'TX', 'WI'}

        assert set(traning_data_df.cat_id.unique()
                   ) == {'HOBBIES', 'HOUSEHOLD', 'FOODS'}

        assert {'d_' + str(n) for n in range(1, 1914)
                }.issubset(traning_data_df.columns)

        assert {'F' + str(n) for n in range(1, 29)
                }.issubset(sample_submission_df.columns)

        assert {'id'}.issubset(sample_submission_df.columns)
        assert traning_data_df.d_1.sum() == 32631
        assert traning_data_df.d_42.sum() == 25572
        assert traning_data_df.d_1913.sum() == 49795
        assert set(traning_data_df.id.unique()
                   ).issubset(sample_submission_df.id.unique())

        
        # now, let's make sure __init__ received correct arguments
        assert n_years_to_benchmark_against in {1, 2, 3, 4, 5}
        assert drop_Xmas_days in {True, False}
        assert n_df_lines_to_print_in_report in range(1, 101)
        assert logging_level in {
            logging.DEBUG, logging.INFO, logging.WARNING}
        
        # finally, actual initialisation
        super().__init__(logger_name="M5",reveal_loggers_identity=False)
        super().update_logger(new_level = logging_level)

        self.n_years = n_years_to_benchmark_against
        self.n_df_lines_to_print_in_report = (
            n_df_lines_to_print_in_report )

        self.sample_submission_df = sample_submission_df
        self.training_df = copy.deepcopy(traning_data_df)

        Xmas_days_list = [
            "d_331", "d_697", "d_1062", "d_1427", "d_1792"]

        if drop_Xmas_days:
            self.training_df.drop(columns=Xmas_days_list
                                  , inplace=True)

        self.training_df["id"] = (
                self.training_df["item_id"].astype("str")
                + "_" + self.training_df["store_id"].astype("str"))

        self.training_df.set_index("id", inplace=True)

        all_training_day_numbers = [int(s[2:])
                                    for s in self.training_df.columns
                                    if s.startswith("d_")]

        n_days_to_benchmark_against = 365 * n_years_to_benchmark_against
        if n_days_to_benchmark_against > len(all_training_day_numbers):
            n_days_to_benchmark_against = len(all_training_day_numbers)

        dates_used_for_benchmark = sorted(
            all_training_day_numbers)[-n_days_to_benchmark_against:]

        self.training_df = (
            self.training_df[["d_" + str(n)
                              for n in dates_used_for_benchmark]
                             + list(self.level_labels_set)])

        self.training_AtL = self._twelve_aggregations(self.training_df)

        self.level_labels_df = (
            copy.deepcopy(self.training_df[self.level_labels_set]))

        self.training_tL = (
            self.training_df.drop(columns=self.level_labels_set))

        del self.training_df

        self.daily_minimums = self.training_AtL.min(axis="columns")
        self.daily_maximums = self.training_AtL.max(axis="columns")

        self.training_monthly_totals_AtL = self.training_AtL.rolling(
            window=28, min_periods=28, axis="columns").sum()

        self.training_monthly_totals_AtL.dropna(
            axis="columns", inplace=True)

        self.monthly_minimums = (
            self.training_monthly_totals_AtL.min(axis="columns"))

        self.monthly_maximums = (
            self.training_monthly_totals_AtL.max(axis="columns"))

        self.no_sales = (
            list(self.training_tL[
                self.training_tL.sum(axis="columns") == 0]
                 .index.unique()))

        training_nonzeros_tL = self.training_tL.astype("bool").astype("int")

        training_nonzeros_rolling_monthly_tL = (
            training_nonzeros_tL.rolling(
                window=28, min_periods=28, axis="columns"
            ).sum())

        training_nonzeros_rolling_monthly_tL.dropna(
            axis="columns", inplace=True)

        historical_monthly_nonzeros = (
            training_nonzeros_rolling_monthly_tL.sum())

        self.time_window_size = 28 * len(self.training_tL)

        self.min_nonzeros = (
            100.0 * historical_monthly_nonzeros.min() 
            / self.time_window_size)

        self.max_nonzeros = (
            100.0 * historical_monthly_nonzeros.max() 
            / self.time_window_size)

    def is_submission_structure_OK(
            self
            , submission_to_check_df):
        """Perfom basic formal check for a submission DataFrame"""

        passed_basic_checks = True

        if not isinstance(submission_to_check_df, type(pd.DataFrame())):
            self.error("Expected Pandas DataFrame.\n")
            return False

        if submission_to_check_df.shape != self.sample_submission_df.shape:
            self.error(f"Submission dataset has a wrong shape"
                + f" {submission_to_check_df.shape}, "
                + f"Expected shape is {self.sample_submission_df.shape}.\n")
            passed_basic_checks = False

        if (set(submission_to_check_df.columns)
                != set(self.sample_submission_df.columns)):

            self.error("Submission dataset has wrong column names.\n")
            passed_basic_checks = False

        else:
            self.info(f"Submission dataset has correct column names"
                + " (check passed).\n")

        if (set(submission_to_check_df.id.unique())
                != set(self.sample_submission_df.id.unique())):

            self.error("Submission dataset has wrong values"
                + " in 'id' columns.\n")

            passed_basic_checks = False

        else:
            self.info(f"Submission dataset has correct values"
                + " in 'id' column (check passed).\n")

        num_of_na_s = submission_to_check_df.isna().sum().sum()

        if num_of_na_s > 0:
            self.error(f"Submission dataset contains"
                + f" {num_of_na_s} N/A-s.\n")
            passed_basic_checks = False
        else:
            self.info(f"Submission dataset does not contain"
                + " N/A-s (check passed).\n")

        num_of_negatives = (
            (submission_to_check_df.drop(columns=["id"]) < 0).sum().sum())

        if num_of_negatives > 0:
            self.error(f"Submission dataset contains"
                + f" {num_of_negatives} negative values.\n")
            passed_basic_checks = False
        else:
            self.info(f"Submission dataset does not contain"
                + " negative values (check passed).\n")

        return passed_basic_checks

    def create_val_eval(
            self
            , submission_to_check_df):
        """Split a submission dataset into Validation and Evaluation"""

        if not self.is_submission_structure_OK(submission_to_check_df):
            return None

        all_ids = list(submission_to_check_df.id.unique())
        validation_ids = [i for i in all_ids if i.endswith("_validation")]
        evaluation_ids = [i for i in all_ids if i.endswith("_evaluation")]
        if not (len(validation_ids) == len(evaluation_ids) == 30490):
            self.error("Submission dataframe contains incorrect id-s")
            return None

        self.info("...splitting a candidate submission dataset"
            + " into Validation and Evaluation sets...\n")

        validation_tL = (
            copy.deepcopy(
                submission_to_check_df[
                    submission_to_check_df.id.isin(validation_ids)]))

        validation_suffix_len = len("_validation")
        validation_tL["id"] = validation_tL["id"].str[:-validation_suffix_len]
        validation_tL.set_index("id", inplace=True)
        validation_tL.columns = [int(c[1:]) for c in validation_tL.columns]
        validation_tL.columns = [
            "d_" + str(1913 + c) for c in validation_tL.columns]

        evaluation_tL = (
            copy.deepcopy(
                submission_to_check_df[
                    submission_to_check_df.id.isin(evaluation_ids)]))

        evaluation_suffix_len = len("_evaluation")
        evaluation_tL["id"] = evaluation_tL["id"].str[:-evaluation_suffix_len]
        evaluation_tL.set_index("id", inplace=True)
        evaluation_tL.columns = [int(c[1:]) for c in evaluation_tL.columns]
        evaluation_tL.columns = [
            "d_" + str(1941 + c) for c in evaluation_tL.columns]

        if not (len(self.training_tL)
                == len(validation_tL)
                == len(evaluation_tL)):
            self.error("Submission file contains wrong # of id-s")
            return None

        if not (set(self.training_tL.index)
                == set(validation_tL.index)
                == set(evaluation_tL.index)):
            self.error("Submission file contains incorrect id-s")
            return None

        ValEval = namedtuple("M5ValEval", ["validation", "evaluation"])

        return ValEval(validation=validation_tL, evaluation=evaluation_tL)

    def _one_aggregation(
            self
            , data_to_aggregate_tL
            , ids_to_group_by_list):
        """Create one (out of 12) level of aggregation"""

        assert len(ids_to_group_by_list) in {1, 2}
        assert set(ids_to_group_by_list).issubset(self.level_labels_set)

        ids_to_drop_set = self.level_labels_set - set(ids_to_group_by_list)

        aggregated_tL = (data_to_aggregate_tL.drop(columns=ids_to_drop_set)
            .groupby(ids_to_group_by_list).sum())

        if len(ids_to_group_by_list) == 2:
            aggregated_tL["line_id"] = (
                aggregated_tL.index.get_level_values(0).astype("str")
                + "_"
                + aggregated_tL.index.get_level_values(1).astype("str"))

            aggregated_tL = aggregated_tL.set_index("line_id")

        else:
            aggregated_tL.index = aggregated_tL.index.astype("str") + "_X"

        return aggregated_tL

    def _twelve_aggregations(
            self
            , input_tL):
        """Create all 12 levels of aggregation"""

        assert self.level_labels_set.issubset(input_tL.columns)

        d_labels_set = {c for c in input_tL.columns
                        if c.startswith("d_")}

        assert (len(d_labels_set) + len(self.level_labels_set)
                == len(input_tL.columns))

        assert len(input_tL) == 30490

        all_aggregations_list = []
        aggregated_totals = copy.deepcopy(input_tL)

        aggregated_totals.drop(
            columns=self.level_labels_set, inplace=True)

        aggregated_totals = pd.DataFrame(aggregated_totals.sum())
        aggregated_totals.columns = ["Total_X"]
        aggregated_totals = aggregated_totals.transpose()
        all_aggregations_list += [aggregated_totals]

        aggregation_fields_list = [['item_id'], ['dept_id'], ['cat_id']
            , ['store_id'], ["state_id"]
            , ['state_id', 'cat_id']
            , ['state_id', 'dept_id']
            , ['store_id', 'cat_id']
            , ['store_id', 'dept_id']
            , ['state_id', 'item_id']
            , ['item_id', 'store_id']]

        for af in aggregation_fields_list:
            all_aggregations_list += [self._one_aggregation(input_tL, af)]

        output_AtL = pd.concat(all_aggregations_list)

        input_cksum = 12 * input_tL[d_labels_set].sum().sum()
        output_cksum = output_AtL.sum().sum()

        assert (abs(input_cksum - output_cksum)
                < (input_cksum + output_cksum) * 0.00000000001)

        return output_AtL

    def check_28day_df(
            self
            , to_check_tL
            , name):
        """Perfom advanced consistency checks for a 28-day DataFrame"""

        assert to_check_tL.shape == (30490, 28)
        assert isinstance(name, str)

        passed_sanity_checks = True

        self.info("<_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_>\n")
        self.info(f"...performing consistency checks"
            + f" for the {name} dataset...\n")

        # let's look at % of non-zero values per 28-day period
        # and benchmark it against historical data.
        # since techically predictions can have non-integer values,
        # we will count any x<0.5 as zero

        current_nonzeros = (100.0 * (to_check_tL >= 0.5).sum().sum()
            / self.time_window_size)

        if not ((self.min_nonzeros - 10)
                < current_nonzeros
                < (self.max_nonzeros + 10)):

            self.warning(f"{name} dataset (no-aggregation version) "
                + f"has {current_nonzeros:.0f}% of non-zero values,"
                + f" while historically there were between"
                + f" {self.min_nonzeros:.0f}% and {self.max_nonzeros:.0f}%"
                + f" non-zeros in any 28-day period in the final"
                + f" {self.n_years} year(s) of the training data\n")

            passed_sanity_checks = False

        else:
            self.info(f"Density of non-zero values in {name} dataset"
                + f" looks OK (check passed).\n")

        # some items-per-store had 0 sales for the final years
        # of the training period; if we forecast non-zero future sales
        # for these items-stores, it will look strange

        if not (to_check_tL[
            to_check_tL.index.isin(self.no_sales)].sum().sum()== 0):

            should_be_no_sales_df = (
                to_check_tL[to_check_tL.index.isin(self.no_sales)
                ].sum(axis="columns"))

            bad_items = (
                list(should_be_no_sales_df[should_be_no_sales_df != 0
                                           ].index.unique()))

            self.warning(f"{len(self.no_sales)} items had zero sales"
                + f" during the final {self.n_years} year(s) of the"
                + f" training period. We assume they were discontinued"
                + f" from distribution in respected stores."
                + f" However, {len(bad_items)} of these items have"
                + f" non-zero sales in original"
                + f" (no aggregation) {name} dataset.\n")

            self.debug("Offending items are:\n" + str(bad_items) + "\n")

            passed_sanity_checks = False

        else:

            self.info(f"{name} dataset does not predict sales for"
                + " discontinued items (check passed).\n")

        # now let's switch our attention from an original sales forecast
        # to an aggegated version

        to_check_AtL = (
            self._twelve_aggregations(self.level_labels_df.join(to_check_tL)))

        # let's check daily max/min item-per-store sales
        # against historical data

        dataset_maximums = to_check_AtL.max(axis="columns")

        above_doubled_maximums = (
                dataset_maximums > 2 * self.daily_minimums)

        num_above_doubled_maximums = above_doubled_maximums.sum()

        if num_above_doubled_maximums > 0:
            self.warning(f"Aggregated {name} dataset contains"
                + f" {num_above_doubled_maximums} daily values"
                + f" which are above doubled daily maximums (calculated for"
                + f" the final {self.n_years} year(s) of training data) \n")

            offences = dataset_maximums[above_doubled_maximums]
            offences.sort_values(ascending=False, inplace=True)
            offences = pd.DataFrame(offences)
            offences.columns = ["Max values"]
            self.debug(f"Some offending examples from"
                + f" aggregated {name} dataset:\n"
                + str(offences.head(self.n_df_lines_to_print_in_report))
                + "\n")

            passed_sanity_checks = False

        else:
            self.info(f"Daily maximums in {name} dataset"
                         + " are OK (check passed).\n")

        # this particular check only makes sense
        # if we have removed Christmas day from the training dataset
        # or if we have performed Christmas imputation,
        # otherwise the check has no impact
        dataset_minimums = to_check_AtL.min(axis="columns")
        below_minimums = (dataset_minimums < self.daily_minimums)
        num_below_minimums = below_minimums.sum()

        if num_below_minimums > 0:
            self.warning(f"Aggregated {name} dataset contains"
                + f" {num_below_minimums} items "
                + f"with values that are below daily minimums "
                + f"(calculated for the final {self.n_years}"
                + f" year(s) of training data) \n")

            offences = dataset_minimums[below_minimums]
            offences.sort_values(inplace=True)
            offences = pd.DataFrame(offences)
            offences.columns = ["Min values"]
            self.debug(f"Some offending examples from"
                + f" aggregated {name} dataset:\n"
                + str(offences.head(self.n_df_lines_to_print_in_report))
                + "\n")

            passed_sanity_checks = False
        else:
            self.info(f"Daily minimums in {name} dataset are"
                         + f" OK (check passed).\n")

        # let's check monthly max/min item-per-store sales
        # against historical data (calculated over rolling 28-day periods)

        monthly_totals = to_check_AtL.sum(axis="columns")
        above_doubled_maximums = (
                self.monthly_maximums * 2 < monthly_totals)

        times_above_doubled_maximums = above_doubled_maximums.sum()

        if times_above_doubled_maximums > 0:
            self.warning(f"Aggregated {name} dataset contains "
                + f"{times_above_doubled_maximums} monthly total values "
                + f"which are above doubled monthly maximums "
                + f"(calculated for the final {self.n_years}"
                + f" year(s) of training data). \n")

            offences = monthly_totals[above_doubled_maximums]
            offences.sort_values(ascending=False, inplace=True)
            offences = pd.DataFrame(offences)
            offences.columns = ["Monthly totals"]
            self.debug(f"Some offending examples from aggregated"
                + f" {name} dataset:\n"
                + str(offences.head(self.n_df_lines_to_print_in_report))
                + "\n")

            passed_sanity_checks = False
        else:
            self.info(f"All monthly totals in {name} dataset"
                + " are below doubled historical"
                + " maximums (check passed).\n")

        below_minimums = (monthly_totals < self.monthly_minimums)
        times_below_minimums = below_minimums.sum()

        if times_below_minimums > 0:
            self.warning(f"Aggregated {name} dataset"
                + f" contains {times_below_minimums} monthly"
                + f"total values which are below monthly minimums"
                + f" (calculated for the final {self.n_years}"
                + f" year(s) of training data. \n")

            offences = monthly_totals[below_minimums]
            offences.sort_values(ascending=False, inplace=True)
            offences = pd.DataFrame(offences)
            offences.columns = ["Monthly totals"]
            self.debug(f"Some offending examples from aggregated"
                + f" {name} dataset:\n"
                + str(offences.head(self.n_df_lines_to_print_in_report))
                + "\n")

            passed_sanity_checks = False
        else:
            self.info(f"All monthly totals in {name} dataset are "
                         + f"above historical minimums (check passed).\n")

        return passed_sanity_checks

    def is_submission_healthy(
            self
            , submission_to_check_df
            , n_df_lines_to_print_in_report="default"):
        """Run all checks for a candidate M5-Acc submission dataframe."""

        if n_df_lines_to_print_in_report != "default":
            assert n_df_lines_to_print_in_report in range(1, 101)

            self.n_df_lines_to_print_in_report = (
                n_df_lines_to_print_in_report)

        self.info("Starting sanity checking process"
            + " for a candidate submission...\n")

        two_datasets = self.create_val_eval(submission_to_check_df)

        if two_datasets is not None:

            val_is_OK = self.check_28day_df(
                two_datasets.validation, "Validation")

            eval_is_OK = self.check_28day_df(
                two_datasets.evaluation, "Evaluation")

            submission_is_OK = (val_is_OK and eval_is_OK)

        else:

            submission_is_OK = False

        self.info("<_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_>\n")

        if submission_is_OK:
            self.info("...sanity check for a candidate submission"
                + " has finished with a good news. The submission"
                + " meets basic criteria and is healthy."
                + " It's OK to proceed and upload "
                + " the submission to Kaggle.")
        else:
            self.warning("...sanity check for a candidate submission"
                + " has finished with a BAD news. The"
                + " submission does not meet basic criteria,"
                + " it is NOT HEALTHY.")

        return submission_is_OK


# In[ ]:


# Here is an example of how M5ACC_SubmissionGatekeeper class should be used
#
# Its constructor expects two main (required) parameters: 
# (1) a pd.DataFrame with original M5 sales_train_validation.csv dataset,
# and (2) another pd.DataFrame with original sample_submission.csv dataset
# 
# there are also three more (optional) parameters:
# (3) n_years_to_benchmark_against indicated the depth of historic data that 
# we wnat to use to generate benchmarks. More is not nessesearly better here 
# because we are mostly collecting daily/monthly min-s/max-s.
# We recommend setting it to 1 or 2 years.
# (4) drop_Xmas_days indicates that we want to 
# exclude Christmas from the analysis.
# While techically Christmas days are present in the training dataset, 
# it appears that the stores were actually closed on these days, 
# as they contain near-zero values.
# (5) df_lines_to_ptint_in_report contains a number of lines we wnat to 
# print while outputing offending datasets in DEBUG mode
# (6) logging_level instructs the Gatekeeper to produce more or less 
# detailed report
#
# Once M5ACC_Submission_Gatekeeper object is created, 
# its method .is_submission_healthy(pd.DataFrame) 
# should be used to validate consistancy of your M5 submission
#
# It only works for "M5 Forecasting - Accuracy" competition
# Submissions to "M5 Forecasting - Uncertainty" competition can NOT be validated
# by M5ACC_Submission_Gatekeeper
#
# Depending on the logging_level (logging.DEBUG, logging.INFO, 
# logging.WARNING, etc.), .is_submission_healthy() can produce 
# more or less detailed output


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Here we create an instance of our sanity / consistency validator\n# The process may take a few dozens of seconds\ngatekeeper = M5ACC_SubmissionGatekeeper(\n    M5_SALES_TRAIN_VALIDATION_DF\n    ,M5_SAMPLE_ACC_SUBMISSION_DF\n    ,n_years_to_benchmark_against = 1\n    ,drop_Xmas_days = True\n    ,logging_level = logging.DEBUG )')


# In[ ]:


# Here we create a submission dataset populated with random numbers
# We will use it to demostrate how the validator works
random_submission_df = copy.deepcopy(M5_SAMPLE_ACC_SUBMISSION_DF)
f_labels = ['F'+str(n) for n in range(1,29)]
random_submission_df[f_labels] = np.random.rand(60980,28)*3


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Now, let\'s use the validator to "diagnose" the submission\nsubmission_is_good = gatekeeper.is_submission_healthy(random_submission_df)')


# In[ ]:


if submission_is_good:
    random_submission_df.to_csv(
        os.path.join(OUTPUT_DATA_DIR,"new_submission_file.csv"))


# In[ ]:


# Now, if it looks interesting to you, scroll up and 
# spend some time reading the code for class M5ACC_SubmissionGatekeeper
# Disregard class LoggableObject as it does not add much value 
# from the perspective of the competition - it is just a small helper tool 

