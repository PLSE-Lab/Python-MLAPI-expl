#!/usr/bin/env python
# coding: utf-8

# # Modeling Price
# 
# What can we learn about AirBnB rentals if we try to model their price? In this notebook we fit a simple linear regression classifier to AirBnB prices for rentals in the Boston market, keeping in mind predictor variables we determined were important in the [Exploring Prices notebook](https://www.kaggle.com/residentmario/d/airbnb/boston/exploring-prices).
# 
# Keep in mind here that our purpose isn't to write a super-accurate classifier of rental prices. We're interested instead in running a simple model that can hopefully give us more insight into what variables matter when it comes to AirBnB rentals, and which ones don't. Linear classification is ideal for this.

# ## Designing Features
# 
# Before we class run our classifier we have to featurize our dataset into something our classifier will understand.

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option("max_columns", None)

listings = pd.read_csv("../input/listings.csv")


# In[ ]:


listings.head()


# In[ ]:


listings['price'] = listings['price'].map(lambda p: int(p[1:-3].replace(",", "")))
listings['amenities'] = listings['amenities'].map(
    lambda amns: "|".join([amn.replace("}", "").replace("{", "").replace('"', "")\
                           for amn in amns.split(",")])
)


# In[ ]:


np.concatenate(listings['amenities'].map(lambda amns: amns.split("|")).values)


# In[ ]:


np.concatenate(listings['amenities'].values)


# In[ ]:


amenities = np.unique(np.concatenate(listings['amenities'].map(lambda amns: amns.split("|")).values))
amenities_matrix = np.array([listings['amenities'].map(lambda amns: amn in amns).values for amn in amenities])


# In[ ]:


amenities_matrix


# To start with, let's fix up two variables with wonky formatting: `price`, saved with dollar signs and commas, and amenities, saved as an even wonkier stringification of a set.

# In[ ]:


listings['price'] = listings['price'].map(lambda p: int(p[1:-3].replace(",", "")))
listings['amenities'] = listings['amenities'].map(
    lambda amns: "|".join([amn.replace("}", "").replace("{", "").replace('"', "")\
                           for amn in amns.split(",")])
)


# In[ ]:


listings.head()


# Let's start off our feature matrix with numerical variables which we can use immediately.

# In[ ]:


features = listings[['host_listings_count', 'host_total_listings_count', 'accommodates', 
                     'bathrooms', 'bedrooms', 'beds', 'price', 'guests_included', 'number_of_reviews',
                     'review_scores_rating']]


# Next, let's turn the presence or absence of the various amenities our AirBnB homes offer into features. We'll encode presence or absence using True and False (which our classifier later on will interpret as 0s and 1s, as these are indeed considered to be equal within Python). To do that we'll map across our amenities lists, assign booleans, concatenate all of the boolean lists into an array, assign that to a proper DataFrame, then concatenate that DataFrame to our existing features matrix. Whoo, that's a mouthful.

# In[ ]:


listings['amenities'].head()


# In[ ]:


listings['amenities'].map(lambda amns: amns.split("|")).head()


# In[ ]:


# The [1:] is because the first amenity our parser finds is a junk empty string---"".
np.unique(np.concatenate(listings['amenities'].map(lambda amns: amns.split("|"))))[1:]


# In[ ]:


amenities = np.unique(np.concatenate(listings['amenities'].map(lambda amns: amns.split("|"))))[1:]
amenity_arr = np.array([listings['amenities'].map(lambda amns: amn in amns) for amn in amenities])
amenity_arr


# In[ ]:


features = pd.concat([features, pd.DataFrame(data=amenity_arr.T, columns=amenities)], axis=1)


# A number of these features are already boolean features, except that they are saved as strings of the form "t" or "f".

# In[ ]:


listings['host_is_superhost'].head()


# We need to fix those too if we want to use them, by mapping them to True and False.

# In[ ]:


for tf_feature in ['host_is_superhost', 'host_identity_verified', 'host_has_profile_pic',
                   'is_location_exact', 'requires_license', 'instant_bookable',
                   'require_guest_profile_picture', 'require_guest_phone_verification']:
    features[tf_feature] = listings[tf_feature].map(lambda s: False if s == "f" else True)


# We also have a number of categorical fields, e.g. `bed_type` which may be any of `{real_bed, futon, sofa, [...]}`.

# In[ ]:


listings['bed_type'].head()


# We'll encode these into dummy variables too, using the built-in `pandas` `get_dummies` convenience function.

# In[ ]:


pd.get_dummies(listings['bed_type']).head()


# In[ ]:


for categorical_feature in ['neighbourhood_cleansed', 'property_type', 'room_type', 'bed_type']:
    features = pd.concat([features, pd.get_dummies(listings[categorical_feature])], axis=1)


# Seems we have everything.

# In[ ]:


features.head()


# Do any of our columns have empty values?

# In[ ]:


for col in features.columns[features.isnull().any()]:
    print(col)


# Our classifier will not like us if we hand it empty values, so we have to fill these in somehow. The median of the columns is a pretty good way of doing it.

# In[ ]:


for col in features.columns[features.isnull().any()]:
    features[col] = features[col].fillna(features[col].median())


# One more thing. Regression can be get thrown off very easily by extreme values, and this dataset has a few really big ones.

# In[ ]:


features['price'].sort_values().reset_index(drop=True).plot()


# Exploring our dataset earlier (in Exploring Price) showed that though the extremely high-value rentals may sometimes be actual mansions or whatnot for rent, for the most part they're scalpers setting unrealistic valuations or just joke listings. These will throw off our model if we include them, by a lot.
# 
# Exploring Price showed that a price ceiling of around 600 per day was reasonable, so let's use that here.

# In[ ]:


fitters = features.query('price <= 600')


# # Classification
# 
# Ok, now we're ready for a classifier. Let's use `scikit` `LinearRegression` (there are two dominant options here, this one and the `statsmodel` one, and we're going to go with `scikit` for no particular reason).

# In[ ]:


from sklearn.linear_model import LinearRegression


# Fit the model and generate predictions.

# In[ ]:


clf = LinearRegression()
y = fitters['price']
clf.fit(fitters.drop('price', axis='columns'), y)


# In[ ]:


y_pred = clf.predict(fitters.drop('price', axis='columns'))


# ## Measuring Performance
# 
# To see how well our classifier did, let's look at some metrics.

# In[ ]:


import sklearn.metrics


# MSE is the square of the average error in each term, while root MSE is its absolute value.

# In[ ]:


mse = sklearn.metrics.mean_squared_error(y, y_pred)
mse


# In[ ]:


root_mse = mse**(1/2)
root_mse


# Our RMSE is 59 dollars, meaning that our classifier is wrong by that much on average.
# 
# How significant is this with respect to the range of prices we are seeing? To see that let's plot RMSE as a boundary around the median price.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# In[ ]:


sns.kdeplot(y)
ax = plt.gca()
ax.set_xlim([0, 600])
plt.axvline(y.median(), c='black')
ax.add_patch(
    patches.Rectangle((y.median() - root_mse, 0), 2*root_mse, 0.006,
                      color='r', alpha=0.2)
)


# That's...not great. A look at our R-squared, which tells us the percentage of the data's variance explained by the model, confirms that.

# In[ ]:


r_squared = sklearn.metrics.r2_score(y, y_pred)
r_squared


# Of course, we knew this ahead of time. Price is a difficult thing to classify given our limited variables and simplistic model worldview. Still, I had hoped for R^2 more along the lines of 75% or better...

# ## Observations
# 
# If we were interested in accuracy we'd definitely go back and try and improve our outcome, however for the purposes of figuring out what the important variables are we'll press on with what we have.
# 
# Let's look at our coefficients:

# In[ ]:


coefs = list(zip(clf.coef_, fitters.drop('price', axis='columns')))
coefs


# Some off-the-cuff observations:
# 
# 1. Right away we see a variable that doesn't matter at all: `host_listings_count` and `host_total_listings_count` (which are really almost one variable, it turns out!).
# 
#    Note that `host_listing_count` *does* have an impact, in the case of professional BnB companies, that effect however is more readily explained by a second-order variable that we will examine later on.
# 2. According to our model, every additional bedroom you want to rent will set you back an additional 35 dollars, while each additional bathroom will cost 11 dollars extra.
# 
#    Considering that this is the core product, these are surprisingly low numbers! In other words, a disproportionate amount of price variation relative to our expectations is explained by factors other than the number of rooms available.
# 
#    Here's a logical rationalization: renters going through AirBnB by and large are people looking for a relatively cheap short-term place to stay, and place less emphasis on the number of rooms available where they are staying than the convenience of and amenities available there.
# 
#    This opinion contrasts with the way that hotel pricing generally works. Hotels charge multiplier premiums for additional rooms!
# 3. Each additional bed is just 3 dollars extra, and each additional accomodation and guest included costs 6. Note that usually hosts charge extra fees for additional guests, a fee that has not been factored into our simply model, so this variable doesn't *really* capture the cost of bringing your friends. Still, these very low numbers hint that splitting a BnB is very cost-efficient.
# 4. Surprisingly, the number of reviews *decreases* price. This isn't a huge effect, just 20 cents a review, literal peanuts price-wise. I interpret the smallness of this effect as meaning that we've done a good job getting rid of outliers: specifically, unrealistically high BnB prices meant to be jokes or set by scalpers, which would obviously *not* have any reviews.
# 5. A host's review rating doesn't matter at all (at least, not for the price they charge). [Relevant XKCD](https://xkcd.com/1098/).

# Let's look at the effect that neighborhood has on price.

# In[ ]:


neighborhoods = np.unique(listings['neighbourhood_cleansed'])
neighborhood_effects = [v for v in coefs if v[1] in neighborhoods]


# In[ ]:


pd.Series(data=[n[0] for n in neighborhood_effects],
          index=[n[1] for n in neighborhood_effects])\
    .sort_values()\
    .plot(kind='bar')


# We see that neighborhood does indeed, unsurprisingly, have a strong effect on the price of a listing. Far-from-the-city-center, demographically "sketchy" neighborhoods cost 40 dollars or less than average to rent in, which as a lifelong commuter honestly sounds like kind of a good deal to me (hey, it's still Boston). Again, 40 dollars a night over the average seems like the premium you pay to be "right there" location-wise.
# 
# And South Boston, a once working-class neighborhood now long and deep on gentrification, is the average-most of the nabes.
# 
# Overall the range in the price premiums per choice of neighborhood can almost make up the average price alone! Put another way, every night in the Bay Village is *almost* two nights in Roslindale, on average.
# 
# Next, let's look at amenities.

# In[ ]:


amenity_effects = [v for v in coefs if v[1] in amenities]


# In[ ]:


pd.Series(data=[n[0] for n in amenity_effects],
          index=[n[1] for n in amenity_effects])\
    .sort_values()\
    .plot(kind='bar')


# Amenities are an interesting case. 
# 
# On a face level, more amenities is always better: every additional amenity available at a location will theoretically be a big plus for people who actually want it and a not-negative for those who don't (you can choose, after all, whether or not they make use of it). Hence on an individual level there's an incentive for a host to list irons, hot tubs, kitchens, the works.
# 
# On the level of AirBnBs at large, however, what a host chooses to list as an amenity is a signal about the "class" of the listing. The most obvious example of this to me is hangers. I do not doubt for a second that every (well, almost every) BnB on the market gives its visitors access to hangers, but it's available to list as an amenity all the same. Even when you don't list hangers, then, it's just *assumed* that you have them; hosts who choose to list them as a "feature" anyway bring up the shade of not having such a rather basic necessity, which actually signals a *lower* price.
# 
# So the effect of amenities in our model should be interpreted not as a fungible good but as a *social signal*. From that perspective, a price penalty for including a kitchen, hangers, or Internet access makes sense, with the magnitude of the effect corresponding with the strength of what that signals about the "class" of the rental.
# 
# With that out of the way, here's my observations:
# 
# 1. Other pet(s) and Washer / Dryer only appear in a very small number of listings, for whatever reason (perhaps this option has been disabled for new listings for a long time now?), and can safely be disregarded.
# 2. A kitchen, free street parking, paid parking off premises, hangers, and Internet are all "essentials", and willingness to advertise them as amenities corresponds with a counter-intuitive and significant *decrease* in price.
# 3. Having a pool is seemingly also penalized, which seems strange. But remember that pools are exceedingly rare inside of city limits; correspondingly having a pool signals that the unit in question is somewhere suburban far from the city center, a fact which seemingly overwhelms the positive benefit of having a pool at all. Note that this effect is also partially/mostly explained by neighborhood!
# 
#    Free Parking on Premises nets a minor penalty for the same reason.
# 4. Having a cat invokes a small price penalty, while having a dog basically doesn't change the price you can charge. Strangely, though, claiming instead that "pets live on this property" corresponds with a small cost bump. Of all of the effects this is the one I understand worst.
# 5. Reasonable people don't need 24-hour check-in, apparently; nor do they need smoking rights.
# 6. An entire range of goods&mdash;dryers, hair blowers, irons, first aid kits&mdash;have no significant effect on price.
# 7. Being wheelchair accessible is a big bonus, presumably because it's still a rare thing and an indicator of a building being of recent build and relatively affluent.
# 8. Breakfast service will set you back about ten bucks!
# 9. An indoor fireplace and a doorman are both pretty obvious social signals, and each will set you back about 20 dollars a night.
# 10. If you're a lister, apparently three easy things you *can* do to increase the price you can charge is provide a washer, A/C, and a TV. The former two cost 10 dollars a night (don't most places have A/C?), while the latter is a whopping 15 dollar a night premium.
# 11. Having access to a gym nets a bonus, but a small one. So does advertising a laptop-friendly work environment.

# There's one more effect I want to drill into.

# In[ ]:


[(-37.530452836818469, 'require_guest_profile_picture'),
 (56.642738981875432, 'require_guest_phone_verification')]


# Interestingly enough, requiring guests to have a profile picture reduces price by almost 40 dollars a night, while requiring verification by phone increases it by more than 55 dollars. Those are big numbers; where are they coming from?
# 
# Let's look at the average characteristics to find out.

# In[ ]:


fitters[fitters['require_guest_phone_verification'] == True].mean()


# Whoa! That `host_listings_count` is wild! What's going on here?

# In[ ]:


fitters[fitters['require_guest_phone_verification'] == True]['host_listings_count']


# Ah. It looks like requiring guest phone verification, a logistical nightmare and rarely used option for most sharing economy listers, is a professional requirement for listers somehow managing large amounts of residences.
# 
# By the way, you may have heard on the news that AirBnB is in legal court battles with many cities, [the city of New York most prominently](http://www.nytimes.com/2016/10/22/technology/new-york-passes-law-airbnb.html), over regulations targeting people renting out non-primary residences. What you are seeing here are these firms directly: companies professionally managing 558-home portfolios.
# 
# This variable just happens to be a proxy for such firms, discovering that they tend to charge a significant premium.

# ## Conclusions
# 
# Even though our model's predictive accuracy is poor, it nevertheless points to a large number of interesting facts about the Boston AirBnB market (and, by analogy, to AirBnB markets worldwide). I wouldn't treat the observations above as verbatim, but rather as interesting directions for further exploration if you are so inclined!
# 
# (Did I mention that [AirBnB has one of the best data teams for a company is size](https://www.airbnb.com/careers/departments/data-science-analytics)?)
# 
# 

# ## Further reading
# 
# * https://www.youtube.com/watch?v=80fZrVMurPM
