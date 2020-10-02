#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install git+https://github.com/lisphilar/covid19-sir#egg=covsirphy


# In[ ]:


from pathlib import Path
import numpy as np 
import pandas as pd 
import covsirphy as cs

def main():
    # Create output directory
    code_path = Path('results')
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get the John Hopkins dataset
    jhu_data = cs.JHUData("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
    
    # Get the Turkish dataset
    tr_data = cs.CountryData("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv", country="Turkey")
    tr_data.set_variables(
        # Specify the column names to read
        date="Last_Update",
        confirmed="Confirmed",
        fatal="Deaths",
        recovered="Recovered",
        province=None
    )
    
    # Replacement of John Hopkins data in Turkey
    jhu_data.replace(tr_data)

    # Get the Population dataset
    pop_data = cs.Population("/kaggle/input/covid19-global-forecasting-locations-population/locations_population.csv")
    pop_data.cleaned().tail()
    
    # Create the scenario
    tr_scenario = cs.Scenario(jhu_data, pop_data, "Turkey")
    
    # Show records
    tr_record_df = tr_scenario.records(
        filename=output_dir.joinpath("tr_records.png"))
    tr_record_df.to_csv(output_dir.joinpath("tr_records.csv"), index=False)
   
    # Show S-R trend
    tr_scenario.trend(filename=output_dir.joinpath("tr_trend.png"))
    
    # Find change points
    tr_scenario.trend(
        n_points=4,
        filename=output_dir.joinpath("tr_change_points.png")
    )
    
    print(tr_scenario.summary())
    
    # Hyperparameter estimation
    tr_scenario.estimate(cs.SIRF)
    
    # Show the history of optimization
    tr_scenario.estimate_history(
        phase="1st", filename=output_dir.joinpath("tr_estimate_history_1st.png")
    )
    
    # Show the accuracy as a figure
    tr_scenario.estimate_accuracy(
        phase="1st", filename=output_dir.joinpath("tr_estimate_accuracy_1st.png")
    )
    
    # Add future phase to main scenario
    tr_scenario.add_phase(name="Main", end_date="01Aug2020")
    tr_scenario.add_phase(name="Main", end_date="31Dec2020")
    tr_scenario.add_phase(name="Main", days=100)
    
    # Add future phase to alternative scenario
    sigma_4th = tr_scenario.get("sigma", phase="4th")
    sigma_6th = sigma_4th * 2
    tr_scenario.add_phase(name="New medicines", end_date="31Dec2020", sigma=sigma_6th)
    tr_scenario.add_phase(name="New medicines", days=100)
    
    # Prediction of the number of cases
    sim_df = tr_scenario.simulate(
        name="Main",
        filename=output_dir.joinpath("tr_simulate.png")
    )
    sim_df.to_csv(output_dir.joinpath("tr_simulate.csv"), index=False)
    
    # Save summary as a CSV file
    summary_df = tr_scenario.summary()
    summary_df.to_csv(output_dir.joinpath("tr_summary.csv"), index=True)

if __name__ == "__main__":
    main()

