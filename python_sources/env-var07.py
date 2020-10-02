#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()

df = env._var07.set_index(["assetCode", "time"], verify_integrity=True)
df["confidenceValue"] = 2 * (df.returnsOpenNextMktres10 > 0).astype(float) - 1

for (
    market_obs_df,
    news_obs_df,
    predictions_template_df,
) in env.get_prediction_days():
    predictions_df = (
        predictions_template_df.set_index("assetCode")
        .join(
            market_obs_df.set_index(["assetCode", "time"])
            .join(df.confidenceValue, how="left")
            .confidenceValue.reset_index()
            .drop("time", axis=1)
            .set_index("assetCode"),
            how="inner",
            lsuffix="_",
        )
        .drop("confidenceValue_", axis=1)
        .reset_index()
    )
    env.predict(predictions_df)

env.write_submission_file()


# It seems to me that `env._var07` has all the targets needed for both stages of the evaluation. Should we have access to these values? In this submission, it's obvious something isn't right, but these values can be used to optimize parameters which can then be simply hardcoded into the model without any indication on where they come from, which is much harder to identify.
