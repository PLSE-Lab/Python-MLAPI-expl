#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival: An experimental analysis of feature effects
# 
# There is no shortage of EDA kernels for the Titanic dataset. As a rule, they (1) detail the features which have a high correlation to survival; (2) propose a few novel features; and then (3) jump to "I submitted it and got these results". (Some kernels hide this last step, and leave you to realize later that their novel features fared worse than the base-line "only women survive" submission.)
# 
# Something not revealed in this process is how many solutions were tried and dropped because they performed poorly on the leaderboard data. This makes it possible (and, IMHO, likely) to manually overfit on the LB data, especially when even one extra "lucky guess" can catapult you many positions on the leaderboard.
# 
# My goal here is to employ what I hope is a more scientific process:
# * Use a classifier that doesn't overfit to the training data, thus minimizing "lucky guesses".
# * Show that each feature proves out on the training set via cross-validation.
# * Avoid using the test set during the training process.
# 
# I'll omit the EDA -- not because it isn't important, but because you've already seen it in lots of other kernels. Just pick your favorite, and pretend it was inserted towards the top of this kernel.
# 
# I'll state up front that the final submission score is good, but not astounding: it's 0.79425. Additional feature engineering can undoubtedly produce better results, but will hopefully follow the same experimental process.

# ## Imports and utilities
# 
# Because I wish to be rigorous about the experiments, I start by setting up an experimental framework that allows testing and evaluating different approaches in isolation and in combination. In particular, I'll make heavy use of sklearn Transformers and Pipelines so that features can be computed "on demand" in a non-destructive manner. The original X_train and y_train DataFrames will never be modified, and the final X_test won't even be loaded until we've finished all our experiments.
# 
# If python code bores you, feel free to just skip to "Finding a stable classifier"

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Kaggle's current docker version throws spurious "FutureWarning: Passing (type, 1) or '1type'..."
# warnings. This disables those messages, while still letting our kernel run properly.
import warnings  
warnings.filterwarnings('ignore')


# ### Read in the data
# We do this early so that we can use it when we are testing our various tools.
# 
# The definition of "cc" will probably be a bit confusing. It dynamically creates an object whose attributes are the column names of our dataset. This lets Jupyter and IPython to do autocompletion, and keeps me from misspeling the column names.

# In[ ]:


Xy_train = pd.read_csv("../input/titanic/train.csv", index_col="PassengerId")

# A completion-friendly collection of column names
class CC:
    def __init__(self, dataframe):
        for col in dataframe.columns: setattr(self, col, col)
cc = CC(Xy_train)
 
X_train = Xy_train.drop(columns=[cc.Survived])
y_train = Xy_train[cc.Survived]


# ### A "generic transformer"
# 
# This class allows us to create custom transformers with minimal code. For most simple transformers, we need simply supply a lambda function which creates a new DataFrame from the input. For transformers which must infer aggregate data, we allow a second "fitter" function which computes a single value to be passed on as a second argument to the main transformation function.

# In[ ]:


class GenericTransformer(BaseEstimator, TransformerMixin):
    """Defines a transformer based on simple lambda functions for fitting and transforming."""
    def __init__(self, transformer, fitter=None):
        self.transformer = transformer
        self.fitter = fitter
    def fit(self, X, y=None):
        self.fit_val = None if self.fitter is None else self.fitter(X)
        return self
    def transform(self, X):
        return self.transformer(X) if self.fit_val is None else self.transformer(X, self.fit_val)


# ### A boring data imputer
# 
# This imputer (built on top of the GenericTransformer) imputes median values for numeric columns and the simple feature "missing" for categorical columns. Most of the time, this is all you need.

# In[ ]:


# Creates an imputer which fills missing categorical values with "missing" and missing numerical 
# values with the median value. By using a "fit" step, we ensure that the median values reflect 
# just the train set and not the test set. This will make a small difference during 
# cross-validation.
default_imputer = GenericTransformer(
    fitter = lambda X: {  # create default_values
        col: "missing" if X[col].dtype == "object" else X[col].median() for col in X.columns},
    transformer = lambda X, default_values: (
        X.assign(**{col: X[col].fillna(default_values[col]) for col in X.columns})),
)

assert not default_imputer.fit_transform(X_train).isna().any().any()


# ### A boring one-hot category encoder
# 
# Since we are transforming our train and test sets separately, we can't just naively call "get_dummies". This would almost definitely produce different columns in the train and test sets. It's a bit more complicated than transforming train and test as a unit, but it's cleaner and safer.

# In[ ]:


class DummiesTransformer(BaseEstimator, TransformerMixin):
    """Transformer which applies the DataFrame 'get_dummies' method in a cv-safe manner."""
    def __init__(self, drop_first=False):
        self.drop_first = drop_first
    def fit_transform(self, x, y=None):
        result = pd.get_dummies(x, drop_first=self.drop_first)
        self.output_cols = result.columns
        return result
    def fit(self, x, y = None):
        self.fit_transform(x, y)
        return self
    def transform(self, x):
        x_dummies = pd.get_dummies(x, drop_first=self.drop_first)
        new_cols = set(x_dummies.columns)
        # Return a new DataSet with exactly the columns found in "fit", zero-filling any that 
        # are missing and dropping extras
        return pd.DataFrame({col: x_dummies[col] if col in new_cols else 0 
                             for col in self.output_cols})
add_dummies = DummiesTransformer()

assert (X_train.dtypes == "object").any()
assert not (add_dummies.fit_transform(X_train).dtypes == "object").any()


# ### A transformer to drop extraneous columns
# 
# In order to isolate just the features we explicitly choose, this transformer drops all of the columns which were in the original X_train, leaving anything our transformers added. Just make sure that your transformers define new column names rather than replacing the originals.

# In[ ]:


# Create a transformer which gets rid of all X_train's original columns, leaving just the new
# ones. This should be invoked before add_dummies.
drop_original = GenericTransformer(lambda X: X.drop(columns=X_train.columns))

assert len(drop_original.fit_transform(X_train).columns) == 0


# ## Finding a stable classifier
# 
# Because this is a very small dataset, it is incredibly easy for classifiers to overfit. This experiment shows the effect of introducing noisy data to various algorithms. The "null" feature set is a constant, which forces classifiers to infer results purely based on target frequencies. "rand" provides random numbers which provide no extra information, and "id" adds in the PassengerId column, which *should* be totally uncorrelated with the target. The "sex" feature *does* give useful information, which legitimately increases the accuray.
# 
# As you can see, all but one classifier give wildly divergent results if you add noisy data to either the null classifier or the sex-based classifier. The noise shouldn't affect the prediction, but the classifiers still dive down a rabbit hole and assign significant meaning to it. The one exception is the RidgeClassifier, which is forced to summarize each feature by a single number, and thus can't cherry-pick special cases. Other linear classifiers will probably have similar stability, but I like the behavior of RidgeClassifier, so I'll stick with it.

# In[ ]:


all_classifiers = [
    RidgeClassifier(random_state=0),
    XGBClassifier(random_state=0),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(n_estimators=10, random_state=0),
    KNeighborsClassifier(),
]
clsf_accumulator = pd.DataFrame()
def test_classifiers(label, transformers):
    for clsf in all_classifiers:
        pipeline = make_pipeline(default_imputer, *transformers, drop_original, add_dummies)
        accuracy = cross_val_score(make_pipeline(pipeline, clsf), X_train, y_train, cv=5).mean()
        clsf_accumulator.loc[label, clsf.__class__.__name__] = accuracy

add_const = GenericTransformer(lambda X: X.assign(CONST=1))
add_random = GenericTransformer(lambda X: X.assign(RAND=np.random.random(X.shape[0])))
add_sex = GenericTransformer(lambda X: X.assign(FEMALE=(X[cc.Sex] == "female").astype("int8")))
add_id = GenericTransformer(lambda X: X.assign(ID=X.index))

test_classifiers("null", [add_const])
test_classifiers("null+rand", [add_const, add_random])
test_classifiers("null+rand+id", [add_const, add_id, add_random])
test_classifiers("sex", [add_sex])
test_classifiers("sex+rand", [add_sex, add_random])
test_classifiers("sex+rand+id", [add_sex, add_random, add_id])

display(clsf_accumulator)


# ## Testing the effects of our favorite features
# 
# With the transformers we created above, we can easily test just the new features we create, either in isolation or in combination. We create a simple test-fixture which accumulates classifier accuracies for each feature we want to try. We'll then run through four features which will be shown to provide significant improvement, and add in a specialized imputer which estimates age based upon the "title" portion of the name. This same test-fixture would let you perform informative experiments with any other features you might propose.
# 
# To increase the drama, we'll add the features in reverse-order by effectiveness. After all, you already know that "Sex" is good, so it's more fun to start with a little mystery.

# In[ ]:


feat_accumulator = pd.DataFrame()
model = RidgeClassifier(random_state=0)

def test_features(label, isolated, combination, prev_label=None, extra_imputers=[]):
    isolated_pipeline = make_pipeline(*extra_imputers, default_imputer, *isolated, 
                                      drop_original, add_dummies, model)
    iso_accuracy = cross_val_score(isolated_pipeline, X_train, y_train, cv=5).mean()
    feat_accumulator.loc[label, "isolated"] = iso_accuracy
    
    combo_pipeline = make_pipeline(*extra_imputers, default_imputer, *combination,
                                   drop_original, add_dummies, model)
    combo_accuracy = cross_val_score(combo_pipeline, X_train, y_train, cv=5).mean()
    feat_accumulator.loc[label, "combination"] = combo_accuracy
    if prev_label is not None:
        old_combo_accuracy = feat_accumulator.loc[prev_label, "combination"]
        feat_accumulator.loc[label, "improvement"] = combo_accuracy - old_combo_accuracy
    display(feat_accumulator)

test_features("null", [add_const], [add_const])    


# ### Feature #1: Age
# 
# Since the effects of age on survival are definitely not linear, we divide passengers into 5 different age groups. The precise choice of bin is arbitrary, but reflects the fact that (at the time of sailing) anyone over 14 was considered an adult. (According to the ticket sellers, anyone over 12 had to pay adult fares, but when applying "women and children first", passengers wouldn't pay attention to that definition.)
# 
# We can see that knowing age is better than nothing, but won't get us very far on the leaderboards.

# In[ ]:


add_age = GenericTransformer(
    lambda X: X.assign(AGE_BINNED = pd.cut(
        X[cc.Age], [0,7,14,35,60,1000],
        labels=["young child","child", "young adult", "adult", "old"])))

test_features("age", [add_age], [add_age], "null")


# ### Feature #2: Passenger class
# 
# Again, we convert this feature to bins, even though it is supplied as a numerical value. There's no reason to believe that the relationship to survival is linear, and the extra feature count won't overload our classifiers.
# 
# This one gives us a respectable improvement in isolation, and an even greater boost when used in combination with age. The cross-dependency between these two features is pretty well-known.

# In[ ]:


add_class = GenericTransformer(lambda X: X.assign(CLASS_BINNED=X[cc.Pclass].astype(str)))

test_features("class", [add_class], [add_age, add_class], "age")


# ### Feature #3: Survival of travelling companions
# 
# This feature is the secret sauce to getting above 77%, and is also a bit of a cheat. It involves creating a feature that explicitly incorporates the target value. It also changes the problem definition from "if the ship crashed, would you survive?" to "given that the ship crashed and we know how your relatives did, can we guess whether you survived?".
# 
# When computing this, you have to be really careful not to include the individual passenger's survival in his own group-survival feature. Instead, we give each passenger a fractional survival value corresponding to the mean survival rate, while including *actual* survival rates for everyone else (in our training set) who was on the same ticket. For obvious reasons, we don't incorporate survival rates from the test set.
# 
# Because we want to do accurate cross-validation, we pretend that we know nothing about the survival of passengers who aren't in the set provided to the "fit" method, even if they are part of the overall training set.
# 
# I adapted this technique from the more complex approach used in ["2nd degree families and majority voting"](https://www.kaggle.com/erikbruin/titanic-2nd-degree-families-and-majority-voting/report) by Erik Bruin.

# In[ ]:


class TicketSurvivalTransformer(BaseEstimator, TransformerMixin):
    """Adds the average survival rate of people on the same ticket.
    
    This is a tricky (and questionable) computation which depends upon having access to the
    target feature, while still accepting that it won't be available when transforming the
    test set. Thus, for each row we pretend that we don't know the survival of the individual,
    and instead give it a fractional value equal to the overall mean survival rate."""
    def __init__(self, xy):
        self.xy = xy
    def fit(self, X, y=None):
        X_with_survival = X.assign(Survived = self.xy.reindex(X.index)[cc.Survived])
        self.mean_survival = X_with_survival[cc.Survived].mean()
        self.group_stats = (X_with_survival.groupby(cc.Ticket)[cc.Survived].agg(["count", "sum"]))
        self.fit_X = X.copy()
        return self
    def transform(self, X):
        X_with_survival = X.assign(Survived = self.xy.reindex(X.index)[cc.Survived])
        group_stats_by_passenger = self.group_stats.reindex(X[cc.Ticket].unique(), fill_value=0)
        X_counts = group_stats_by_passenger.loc[X[cc.Ticket]].set_index(X.index)
        is_overlap = np.array([x in self.fit_X.index for x in X.index])
        other_counts = (X_counts["count"]-1).where(is_overlap, X_counts["count"])
        other_survivor_count = ((X_counts["sum"] - self.xy[cc.Survived].reindex(X.index))
                                .where(is_overlap, X_counts["sum"]))
        survival_fraction = (other_survivor_count + self.mean_survival) / (other_counts + 1)
        return X.assign(SurvivorFraction=survival_fraction)

# This assertion is a weak test, but make sure that average group survival is at least close
# to the overall survival rate.
assert np.isclose((TicketSurvivalTransformer(Xy_train).fit(X_train.loc[:400])
                   .transform(X_train.loc[400:])["SurvivorFraction"].mean()),
                  Xy_train.loc[400:][cc.Survived].mean(), rtol=0.1)


# As predicted above, this has a huge effect when applied in isolation. However, the effect in combination with passenger class is less. This is because all passengers on the same ticket will belong to the same passenger class, and the group survival rate is correlated to their class.
# 
# The effect is still enough to give a big boost on the leaderboard, though.

# In[ ]:


add_tkt = TicketSurvivalTransformer(Xy_train)

test_features("tkt", [add_tkt], [add_age, add_class, add_tkt], "class")


# ### Feature #4: Sex
# 
# This one is a no-brainer. After all, guessing "Females survive" will already put you in the top 50% on the leaderboard without any other work or feature engineering. It's worth a solid 17% in isolation and 11% in combination with other features.

# In[ ]:


# We use the add_sex transformer defined in "Finding a stable classifier"

test_features("sex", [add_sex], [add_age, add_class, add_tkt, add_sex], "tkt")


# ### Bonus: better age imputation
#     
# This is a well known technique. Every passenger's name includes a title, and it's a great predictor for both age and gender. "Master" for example, always indicates a boy -- typically quite young -- and "Miss" typically indicates a relatively young woman, though there is no actual upper limit to her age. By using the median age for each title, we can get much more plausible guesses than "everyone is 28 unless stated otherwise", which is the default imputation.
# 

# In[ ]:


class TitleAgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self): pass
    def fit(self, X, y=None):
        self.overall_median = X[cc.Age].median()
        title = X[cc.Name].str.replace(r"[^,]+, *([^.]+)\..*", r"\1")
        self.title_median = X.assign(Title=title).groupby("Title")[cc.Age].agg("median")
        return self
    def transform(self, X):
        title = X[cc.Name].str.replace(r"[^,]+, *([^.]+)\..*", r"\1")
        imputed_ages = self.title_median.reindex(index=title, fill_value=self.overall_median)
        imputed_ages.index = X.index
        result = X.assign(Age=X[cc.Age].fillna(imputed_ages))
        return result

# The average age after imputation should be close to the average supplied age.
assert np.isclose(TitleAgeImputer().fit_transform(X_train)[cc.Age].mean(),
                  X_train[cc.Age].mean(), rtol=2)


# The experimental results are disappointing. The improvement is so small that it's indistinguishable from random noise. However, the technique still seems plausible -- though not foolproof -- so we'll go ahead and use it. (Actual tests against the leaderboard tell us that we get one extra "lucky guess" out of it, but there's no guarantee that it wouldn't suffer equally bad luck when running on the phase 2 test set.)

# In[ ]:


impute_age = TitleAgeImputer()

test_features("age_from_title", [add_age], [add_age, add_class, add_tkt, add_sex], "sex",
              extra_imputers=[impute_age])


# ## Relative feature importance
# 
# By looking at the raw coefficents used by the RidgeClassifier, we can see how much weight is given to each feature. This should, however, be taken with a grain of salt because there isn't a single unique set of coefficients that works for a particular classifier training run. Other runs (not included here) with effectively the same features give a noticably different ordering, with "FEMALE" being #1 instead of demoted to second rank. We could get around this by using a *truly* stable classifier like Naive Bayes, but we would also get much worse results.
# 
# Still, the coefficients reflect our intuition. Females have high positive coefficients, as do ticket groups with high survival rates. Third class passengers and the elderly both have strong negative coefficients which accurately reflects the fact that both groups were effectively doomed.

# In[ ]:


def feature_weights(transformer, X, y):
    transformed = transformer.fit_transform(X)
    model = RidgeClassifier(random_state=0).fit(transformed, y)
    coef = model.coef_[0]    # RidgeClassifier wraps the coefficients in a 1xN array
    coefficients = pd.DataFrame({"feature": transformed.columns,
                                 "coefficient": coef, 
                                 "abs": np.abs(coef)}) 
    sorted = coefficients.sort_values("abs", ascending=False)
    ranked = sorted.assign(Rank=range(1, len(transformed.columns)+1))
    ranked = ranked.append(pd.Series([" ", "**intercept**", model.intercept_[0], 0], name=0,
                                    index=["Rank", "feature", "coefficient", "abs"]))
    return ranked.set_index("Rank", drop=True).drop(columns=["abs"])

final_pipeline = make_pipeline(impute_age, default_imputer, add_age, add_class, add_tkt, add_sex, 
                               drop_original, add_dummies)
display(feature_weights(final_pipeline, X_train, y_train))


# ## Final submission
# 
# We do one final cross-validation with our final feature pipeline, just to make sure that we didn't mess something up. Having confirmed that the numbers are comparable when run on a slightly different cross-validation split, we go ahead and create our final predictions and drop them into 'submissions.csv'.
# 
# As noted in the preface, the final accuracy for our trained classifier is 0.79425. It's noticably lower than our cross-validated guess, but the leaderboard test set is known to run a few points lower than the training set. This is just the natural effect of performing a random split.

# In[ ]:


final_model = RidgeClassifier(random_state=0)
final_X_train = final_pipeline.fit_transform(X_train)

cvscore = cross_val_score(final_model, final_X_train, y_train, cv=4)
print(f"cv = {np.mean(cvscore)}: \n  {list(cvscore)}")
final_model.fit(final_X_train, y_train)

X_test = pd.read_csv("../input/titanic/test.csv", index_col="PassengerId")
final_X_test = final_pipeline.transform(X_test)
preds_test = final_model.predict(final_X_test)
output = pd.DataFrame({'PassengerID': X_test.index,
                       'Survived': preds_test})
file = 'submission.csv'
output.to_csv(file, index=False)
print(f"Wrote predictions to '{file}'")


# ## Conclusion
# 
# Thanks for reading all the way through, or at least caring enough to skip to the back and see how it ends. Hopefully, I've made a convincing argument that rigorous experimentation helps out in judging what really works, and that the predictability of a stable classifier can be a better alternative to a strong classifier with a tendency to overfit.
# 
# Feel free to extend these experiments to try out new features or to see if you can get more lucky guesses with a different classifier algorithm. If you appreciate the insights, please give the kernel an upvote. It won't buy me a cup of coffee, but it'll still give me a warm feeling.
