#!/usr/bin/env python
# coding: utf-8

# There's 25 columns in this dataset but I can't find documentation for them. This notebook aims to try to understand these columns.

# In[2]:


import pandas as pd


# `train.csv`, `validation.csv`, `test.csv` all have the same columns. There's no hidden target variables like you would see in kaggle competitions. 
# 
# Looks like they just did the splitting for us.

# In[63]:


train = pd.read_csv("../input/train.csv")
valid = pd.read_csv("../input/validation.csv")
test  = pd.read_csv("../input/test.csv")

whole = pd.concat([train, valid, test]).reset_index()


# ## \_id
# 
# You'd expect this to be a unique identifier for the row and being an integer does make sense:

# In[4]:


whole["_id"].head()


# But then there are repeated `_id` so it's not a unique identifier for the row?

# In[5]:


print("Number of rows:", len(whole))
print("Number of _ids:", len(whole["_id"].unique()))


# Scrolling down, you'll see that the `_id` becomes datetime. huh?

# In[8]:


whole.iloc[6279:, :].head()


# But also:
# 
# * `_started_at`, a datetime, becomes a boolean
# * `_canary`, whatever that is, which is usually either blank or boolean, is now an integer that looks like an id
# 
# So, there must be an error in the dataset where those 3 columns got in a 3-way mix up. Fixing it:

# In[7]:


is_faulty = whole["_canary"].str.isnumeric().fillna(False)

fix = pd.DataFrame({
    "_canary":     whole[is_faulty]["_started_at"],
    "_id":         whole[is_faulty]["_canary"],
    "_started_at": whole[is_faulty]["_id"]
})

whole.loc[is_faulty, ["_canary", "_id", "_started_at"]] = fix
whole.loc[is_faulty, ["_canary", "_id", "_started_at"]].head()


# Now, everything is right with the world.

# In[9]:


print("Number of rows: ", len(whole))
print("Number of _ids: ", len(whole["_id"].unique()))


# Also, the past is past.

# In[10]:


was_faulty = is_faulty
del is_faulty


# I will be outputting this fix below so you can use it instead of doing this fix in your notebooks.

# ## \_unit_id / sentence
# 
# If `_id` is id, what is `_unit_id`? What is a unit?
# 
# After a quick look of the dataset, looks like a sentence is repeated on multiple rows each of which is a different annotator annotating the sentence. And it looks as if `_unit_id` is a unique identifier for the sentence (which makes me which it was just named `_sentence_id`).

# In[11]:


print("Number of unique      _id:", len(whole["_id"].unique()))
print("Number of unique _unit_id:", len(whole["_unit_id"].unique()))
print("Number of unique sentence:", len(whole["sentence"].unique()))


# Not the same number of unique `_unit_id` and `sentence`? Could this be another error in the dataset?
# 
# Or, I don't know, maybe the same sentence is used in multiple units? (what's a unit?)
# 
# First, let's check if a repeated `_unit_id` always refer to the same `sentence`:

# In[12]:


unique_unit_sentence_pairs = whole[["_unit_id", "sentence"]].drop_duplicates()
print("Number of unique unit_id-sentence pairs: ", len(unique_unit_sentence_pairs))
print("Number of unique unit_id:  ", len(unique_unit_sentence_pairs["_unit_id"].unique()))
print("Number of unique sentences:", len(unique_unit_sentence_pairs["sentence"].unique()))


# Since these numbers match the above ones, then the same sentence really is used in multiple units and that rows with the same `_unit_id` will have the same `sentence` value. (Until now, I'm still trying to convince myself that this is a sound argument)
# 
# Anways, I guess it makes sense `_unit_id` wasn't named `sentence_id`.

# ## sent_id
# 
# Yet another id!
# 
# Does sent here mean transmission and this column is the `id` for that? ..Or this is the `sentence_id` that I'm looking for!

# In[13]:


print("Number of unique sent_id:", len(whole["sent_id"].unique()))


# Each unit has a unique `sent_id`

# In[14]:


unique_unit_sent_pairs = whole[["_unit_id", "sent_id"]].drop_duplicates()
len(unique_unit_sent_pairs)


# Well it definetely isn't a `sentence_id`.

# In[16]:


unique_sentid_sentence_pairs = whole[["sent_id", "sentence"]].drop_duplicates()

has_many_sent_id   = unique_sentid_sentence_pairs["sentence"].value_counts() > 1
has_many_sentences = unique_sentid_sentence_pairs["sent_id" ].value_counts() > 1

num_sentences_with_many_sent_id = has_many_sent_id.value_counts()[True]
num_sent_id_with_many_sentences = has_many_sentences.value_counts()[True]

print("Sentence can have many sent_id: ", num_sentences_with_many_sent_id > 0)
print("sent_id can have many sentences:", num_sent_id_with_many_sentences > 0)


# Still don't know what that is.

# ## term1 / term2
# 
# They are two terms that appear in the sentence, as refered to in the dataset overview.

# ## Bananas in Pyjamas
# 
# Here, we see that `b1` and `b2` are just the index location of `term1` and `term2` in `sentence`, respectively.

# In[20]:


def get_b1_and_b2(row):
    return (
        row["sentence"].find(row["term1"]) == row["b1"],
        row["sentence"].find(row["term2"]) == row["b2"],
    )

whole.apply(get_b1_and_b2, axis=1).all()


# ## e1 / e2
# 
# Appears to be similar in structure to b1/b2 but what is this refering to?

# In[64]:


diff_values = pd.DataFrame(whole[["e1", "e2"]].values - whole[["b1", "b2"]].values)
diff_values.mean()


# `e1` and `e2` is always greater than `b1` and `b2` and by about the values you see above.
# 
# Now at this point I really couldn't figure it out so I just guessed "e" stands for "Easter Egg" and that this is some hidden puzzle. I started to prepare a regression model and then it hit me - I'm an idiot.
# 
# "b" means "beginning", not banana; and "e" means "ending", not egg. The differences I calculated above, plus 1, should equal the lengths of `term1` and `term2`.
# 

# In[65]:


term_lengths = whole[["term1", "term2"]].applymap(len).rename(columns=lambda name: name + "_len")
term_diff_eq = (diff_values.as_matrix() + 1) == term_lengths.as_matrix()
term_diff_eq.all()


# Oh, it doesn't. That's disappointing.
# 
# Let's investigate what's going on:

# In[66]:


term_diff_eq = pd.DataFrame(term_diff_eq)
term_diff_eq.apply(pd.Series.value_counts)


# We have a lot of matches of `term1`(column 0) but it's still not perfect. `term2` (column 1) looks a lot worse.
# 
# Let's view a row where there was mismatch in `e2 - b2 + 1` and `len(term2)`:

# In[69]:


e1_e2_error = pd.concat([
    diff_values,
    term_lengths,
    whole[["b1", "b2", "e1", "e2", "term1", "term2", "sentence"]]
], axis=1)[~term_diff_eq[1]]
e1_e2_error.head()


# In[68]:


e1_e2_error.iloc[0]["sentence"]


# In[71]:


e1_e2_error.iloc[0]["term2"]


# `len(term2)` is correct:

# In[72]:


len(e1_e2_error.iloc[0]["term2"])


# `b2` is fine:

# In[73]:


e1_e2_error.iloc[0]["sentence"].find(e1_e2_error.iloc[0]["term2"])


# `e2` must be the problem then. And indeed it is:

# In[74]:


pd.DataFrame({
    "letter": list("IM CEFTRIAXONE"),
    "index":  range(128, 142)
})


# `e2` should be 141, not 142. There's an off-by-one error here.
# 
# Since we did get 336 matches, we know  for sure that there are times that the `b2`, `e2`, and `term2` values are as expected. More so for `b1`, `e1`, and `term1` where we matched 15911 rows correctly.
# 
# This dataset's `e1` `e2` values need fixing.

# ## relation / twrex
# 
# They look to be, ahem, related.

# In[41]:


whole["relation"].value_counts()


# In[42]:


whole["twrex"].value_counts()


# Looks like 2 different answers by 2 different sources. Maybe twrex (which sounds like a dinosaur) is a program to determine that relation and it's full name is `The Wasp: Relation EXtraction`?

# In[43]:


unique_relation_twrex_pairs = whole[["relation", "twrex"]].drop_duplicates()
unique_relation_twrex_pairs.head()


# Each unit has one value for `relation` and `twrex`. That is, multiple rows with the same `_unit_id` would also have the same `relation` and `twrex`.

# In[44]:


print(len(whole[["_unit_id",    "twrex"]].drop_duplicates()))
print(len(whole[["_unit_id", "relation"]].drop_duplicates()))


# ## direction
# 
# direction is just either:
# 
# * `term1` + " " + `relation` + " " + `term2`
# 
# or
# 
# * `term2` + " " + `relation` + " " + `term1`

# In[45]:


whole[["term1", "term2", "relation", "direction"]].head()


# except when it's not and it just says "no_relation" instead.

# In[46]:


whole.loc[whole["direction"] == "no_relation" ,["term1", "term2", "relation", "direction"]].head()


# I'm guessing `relation` is the correct answer here and direction is what the annotator answered?

# ## direction_gold
# 
# It's all blank. Maybe you need a gold account to view this direction value?

# In[47]:


whole["direction_gold"].value_counts()


# ## Other columns (that don't deserve their own section)
# 
# Note that I'm just making educated guesses here
# 
# * `_worker_id` - identifies the annotator for this row
# * `_country` - country code of the worker
# * `_region`- region of the worker
# * `_city` - city of the worker
# * `_ip` - ip address worker used when working on this row
# * `_channel` - company the worker is affliated with; there are 38 of them
# * `_trust` - trustworthyness of the worker? seems a bit judgy
# * `_started_at` - time something started at
# * `_created_at` - a few seconds to a minute after something started
# * `relex_relcos` - looks like a probability; confidence for the reported relation?

# ## 3-way swap fix
# 
# Here's the fixed version of the dataset as I noted above:

# In[75]:


new_train = whole[           : len(train)]
new_valid = whole[len(train) : len(train) + len(valid)]
new_test  = whole[len(train) + len(valid) :]

new_train.to_csv("train.csv")
new_valid.to_csv("validation.csv")
new_test.to_csv( "test.csv")


# Though maybe instead of using the output of this notebook, we can update the dataset instead. What do you think @Kevin Mader? Oh right, tagging doesn't work here and why would it.

# ## Conclusion
# 
# A dataset not having documentation can provide enough entertainment for the whole evening. Guess I'll postpone playing civ5 to tomorrow night.
# 
# This exploration has helped make sense of some of the columns but there are still those that remain a mystery. We'll need more information on those if we are to use them effectively.
# 
# Errors in the dataset were spotted. One was rectified with the output above. The issue with term indexes remain unresolved but I don't think this one is that big a deal.
# 
# In any case, I'll be editing the column metadata in a while to reflect what I learned in this notebook.
