#!/usr/bin/env python
# coding: utf-8

# Attribute Information:
# > 1. age
# > 2. sex
# > 3. chest pain type (4 values)
# > 4. resting blood pressure
# > 5. serum cholestoral in mg/dl
# > 6. fasting blood sugar > 120 mg/dl
# > 7. resting electrocardiographic results (values 0,1,2)
# > 8. maximum heart rate achieved
# > 9. exercise induced angina
# > 10. oldpeak = ST depression induced by exercise relative to rest
# > 11. the slope of the peak exercise ST segment
# > 12. number of major vessels (0-3) colored by flourosopy
# > 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

# In[ ]:


import os
import numpy as np
import pandas as pd
from itertools import product


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, silhouette_score

PATH = "../input/heart-disease-uci/"


# In[ ]:


get_ipython().system('ls ../input/heart-disease-uci')


# In[ ]:


from bokeh.io import output_file, show, output_notebook, push_notebook
from bokeh.plotting import figure, reset_output
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper, CDSView, GroupFilter, FactorRange, Slope
from bokeh.layouts import row,column, gridplot
from bokeh.models.widgets import Tabs, Panel
from bokeh.io import curdoc
from bokeh.core.properties import value

curdoc().theme = 'light_minimal'

output_notebook()


# In[ ]:


df = pd.read_csv(f"{PATH}heart.csv")


# In[ ]:


df.head()


# # Correlation Matrix

# In[ ]:


corr = df.corr()
corr.style.background_gradient(cmap='seismic')


# In[ ]:


df.info()


# # Pandas Profiling

# In[ ]:


import pandas_profiling
df.profile_report(style={'full_width':True})


# In[ ]:


map_gender_to_string = {1: "Male", 0: "Female"}
cat_df = df.copy()
cat_df["sex"] = df["sex"].apply(lambda x: map_gender_to_string[x])


# In[ ]:


map_cp_to_string = {i: f"Type{chr(i)}" for i in range(65, 69)}
cat_df["cp"] = df["cp"].apply(lambda x: map_cp_to_string[x+65])


# In[ ]:


map_fbs_to_string = {1: "High", 0: "Low"}
cat_df["fbs"] = df["fbs"].apply(lambda x: map_fbs_to_string[x])


# In[ ]:


map_target_to_string = {0: "NotHealthy", 1: "Healthy"}
cat_df["target"] = df["target"].apply(lambda x: map_target_to_string[x])


# In[ ]:


source = ColumnDataSource(cat_df)


# In[ ]:


def scatter_plot_with_cat(source, views, colors, legends, hover, title, x, y):
    plot = figure(tools=[hover, "crosshair", "pan", "wheel_zoom", "box_zoom", "reset"], title=title)
    
    for v, c, l in zip(views, colors, legends):
        plot.circle(x=x, y=y, source=source, view=v, color=c, legend=l, muted_alpha=0.1)
    plot.legend.click_policy="mute"
    return plot


# In[ ]:


def get_regression_line(df, x_attr, y_attr, line_color="red"):
    linear_male = LinearRegression().fit(df[x_attr].values.reshape(-1, 1), df[y_attr])
    slope = linear_male.coef_[0]
    intercept = linear_male.intercept_
    regression_line = Slope(gradient=slope, y_intercept=intercept, line_color=line_color)
    return regression_line


# In[ ]:


def scatter(source, df, views, colors, legends, hover, title, x_axis, y_axis):
    regression_line = get_regression_line(df, x_axis, y_axis)
    
    plot = scatter_plot_with_cat(source, views, colors, legends, hover, title, x=x_axis, y=y_axis)
    plot.add_layout(regression_line)
    
    return plot


# # Scatter Plot segregated by Gender 

# In[ ]:


male_view = CDSView(source=source,
                       filters=[GroupFilter(column_name='sex', group="Male")])
female_view = CDSView(source=source,
                        filters=[GroupFilter(column_name='sex', group="Female")])


# In[ ]:


hover = HoverTool(tooltips = [("Resting Blood Pressure","@trestbps"),("Serum Cholestrol","@chol")])

plot1 = scatter(source, df, views=[male_view, female_view], colors=["red", "blue"], legends=["Male", "Female"], hover=hover, title="BPS vs CHOL w. Gender", x_axis="trestbps", y_axis="chol")


# In[ ]:


hover = HoverTool(tooltips = [("Resting Blood Pressure","@trestbps"),("Serum Cholestrol","@thalach")])

plot2 = scatter(source, df, views=[male_view, female_view], colors=["red", "blue"], legends=["Male", "Female"], hover=hover, title="BPS vs THAC w. Gender", x_axis="trestbps", y_axis="thalach")


# In[ ]:


hover = HoverTool(tooltips = [("Age","@age"),("Max Heart Rate","@thalach")])

plot3 = scatter(source, df, views=[male_view, female_view], colors=["red", "blue"], legends=["Male", "Female"], hover=hover, title="AGE vs THAC w. Gender", x_axis="age", y_axis="thalach")


# In[ ]:


hover = HoverTool(tooltips = [("Slope","@slope"),("Old Peak","@oldpeak")])

plot4 = scatter(source, df, views=[male_view, female_view], colors=["red", "blue"], legends=["Male", "Female"], hover=hover, title="SLOPE vs OLDPEAK w. Gender", x_axis="slope", y_axis="oldpeak")


# click on the legend

# In[ ]:


tab1 = Panel(child = plot1,title = "bps vs chol")
tab2 = Panel(child = plot2,title = "bps vs thalach")
tab3 = Panel(child = plot3,title = "age vs thalach")
tab4 = Panel(child = plot4,title = "slope vs oldpeak")
tabs = Tabs(tabs=[tab1, tab2, tab3, tab4])
show(tabs)


# # Scatter Plot segregated by Chest Pain

# In[ ]:


cp1_view = CDSView(source=source,
                       filters=[GroupFilter(column_name='cp', group="TypeA")])
cp2_view = CDSView(source=source,
                        filters=[GroupFilter(column_name='cp', group="TypeB")])
cp3_view = CDSView(source=source,
                       filters=[GroupFilter(column_name='cp', group="TypeC")])
cp4_view = CDSView(source=source,
                        filters=[GroupFilter(column_name='cp', group="TypeD")])


# In[ ]:


hover = HoverTool(tooltips = [("BPS","@trestbps"),("CHOL","@chol")])

plot1 = scatter(source, df, views=[cp1_view, cp2_view, cp3_view, cp4_view], colors=["red", "blue", "green", "yellow"],
                legends=["TypeA", "TypeB", "TypeC", "TypeD"], hover=hover, title="BPS vs CHOL w. CP", x_axis="trestbps", y_axis="chol")


# In[ ]:


hover = HoverTool(tooltips = [("BPS","@trestbps"),("THALACH","@thalach")])

plot2 = scatter(source, df, views=[cp1_view, cp2_view, cp3_view, cp4_view], colors=["red", "blue", "green", "yellow"],
                legends=["TypeA", "TypeB", "TypeC", "TypeD"], hover=hover, title="BPS vs THALACH w. CP", x_axis="trestbps", y_axis="thalach")


# In[ ]:


hover = HoverTool(tooltips = [("AGE","@age"),("THALACH","@thalach")])

plot3 = scatter(source, df, views=[cp1_view, cp2_view, cp3_view, cp4_view], colors=["red", "blue", "green", "yellow"],
                legends=["TypeA", "TypeB", "TypeC", "TypeD"], hover=hover, title="AGE vs THALACH w. CP", x_axis="age", y_axis="thalach")


# In[ ]:


hover = HoverTool(tooltips = [("Slope","@slope"),("Old Peak","@oldpeak")])

plot4 = scatter(source, df, views=[cp1_view, cp2_view, cp3_view, cp4_view], colors=["red", "blue", "green", "yellow"],
                legends=["TypeA", "TypeB", "TypeC", "TypeD"], hover=hover, title="SLOPE vs OLDPEAK w. CP", x_axis="slope", y_axis="oldpeak")


# click on the legend

# In[ ]:


tab1 = Panel(child = plot1,title = "bps vs chol")
tab2 = Panel(child = plot2,title = "bps vs thalach")
tab3 = Panel(child = plot3,title = "age vs thalach")
tab4 = Panel(child = plot4,title = "slope vs oldpeak")
tabs = Tabs(tabs=[tab1, tab2, tab3, tab4])
show(tabs)


# In[ ]:


def partition_hist(df, start_val, end_val, step_val, categories, select_col, groupby_col):
    separated_df = {c: [] for c in categories}
    for val in range(start_val, end_val, step_val):
        if step_val > 1:
            part_df = df[(val <= df[select_col]) & (df[select_col] < (val + step_val))]
        else:
            part_df = df[df[select_col]==val]
        count_df = part_df[[select_col, groupby_col]].groupby(groupby_col).count()
        for c in categories:
            if c in count_df.index:
                separated_df[c].append(count_df.loc[c][0])
            else:
                separated_df[c].append(0)
    return separated_df


# In[ ]:


def plot_hist_cat(separated_df, start_val, end_val, step_val, categories, hover, title, colors, factors):
    source = ColumnDataSource(data=dict(
        x=factors,
        **{c: separated_df[c] for c in categories}
    ))
    
    p = figure(x_range=FactorRange(*factors), tools=[hover], title=title)

    p.vbar_stack(categories, x='x', width=0.9, alpha=0.5, color=colors, source=source,
                 legend=[value(x) for x in categories])

    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "vertical"
    return p


# In[ ]:


def hist(cat_df, values, categories, hover, title, colors, x_axis, y_axis):
    start_val, end_val, step_val = values
    factors = [f"{s}-{s+step_val}" for s in range(start_val, end_val, step_val)]
    
    separated_df = partition_hist(cat_df, start_val, end_val, step_val, categories, x_axis, y_axis)
    plot = plot_hist_cat(separated_df, start_val, end_val, step_val, categories, hover, title, colors, factors)
    
    return plot


# # Histogram bins separated by Age

# In[ ]:


start_val, end_val, step_val = 0, 90, 10 # age bin values

categories = ["Male", "Female"]
hover = HoverTool(tooltips = [("Male","@Male"),("Female","@Female")])

plot1 = hist(cat_df, values=(start_val, end_val, step_val), categories=categories, hover=hover,
             title="Age Segregation for Gender", colors=["red", "blue"], x_axis="age", y_axis="sex")


# In[ ]:


start_val, end_val, step_val = 0, 90, 10 # age bin values

categories = [f"Type{chr(i)}" for i in range(65, 69)]
hover = HoverTool(tooltips = [(f"Type{chr(i)}", f"@Type{chr(i)}") for i in range(65, 69)])
                  
plot2 = hist(cat_df, values=(start_val, end_val, step_val), categories=categories, hover=hover,
             title="Age Segregation for Chest Pain Type", colors=["red", "blue", "green", "yellow"], x_axis="age", y_axis="cp")


# In[ ]:


start_val, end_val, step_val = 0, 90, 10 # age bin values

categories = ["High", "Low"]
hover = HoverTool(tooltips = [("High", "@High"), ("Low", "@Low")])

plot3 = hist(cat_df, values=(start_val, end_val, step_val), categories=categories, hover=hover,
             title="Age Segregation for Blood Sugar", colors=["red", "blue"], x_axis="age", y_axis="fbs")


# In[ ]:


start_val, end_val, step_val = 0, 90, 10 # age bin values

categories = ["NotHealthy", "Healthy"]
hover = HoverTool(tooltips = [("NotHealthy", "@NotHealthy"), ("Healthy", "@Healthy")])

plot4 = hist(cat_df, values=(start_val, end_val, step_val), categories=categories, hover=hover,
             title="Age Segregation for Target", colors=["red", "blue"], x_axis="age", y_axis="target")


# In[ ]:


plot1.x_range = plot2.x_range = plot3.x_range = plot4.x_range # equal width
plot1.y_range = plot2.y_range = plot3.y_range = plot4.y_range # equal height


# In[ ]:


tab1 = Panel(child = plot1, title = "Sex")
tab2 = Panel(child = plot2, title = "Chest Pain")
tab3 = Panel(child = plot3, title = "Fasting Blood Sugar")
tab4 = Panel(child = plot4, title = "Target")

tabs = Tabs(tabs=[tab1, tab2, tab3, tab4])
show(tabs)


# # PCA

# In[ ]:


pca = PCA(n_components=13)


# In[ ]:


x_orig = df.drop("target", axis=1).values 
y = df["target"]


# In[ ]:


scaler = StandardScaler()
x_orig_norm = scaler.fit_transform(x_orig)

x_pca_orig = pca.fit_transform(x_orig_norm)


# ## Scree Plot

# In[ ]:


eigen_values = pca.explained_variance_ratio_[:5]
cumulative_value = np.cumsum(eigen_values)
x_label = [f"Component{i}" for i in range(len(eigen_values))]

plot = figure(x_range=x_label)
plot.vbar(x=x_label, top=eigen_values, width=0.9)
plot.line(x=x_label, y=cumulative_value, 
         color='red', line_width=1,
         legend='Cumulative')
plot.xaxis.major_label_orientation = "vertical"
plot.legend.location = "top_left"
show(plot)


# In[ ]:


dimension = 2
x_pca = x_pca_orig[:, :dimension]


# In[ ]:


map_target = {1: "red", 0: "blue"}


# In[ ]:


scatter_df = pd.DataFrame({"x_comp_0": x_pca[:, 0], "x_comp_1": x_pca[:, 1], 
                           "color": [map_target[c] for c in df["target"]], 
                           "label": [map_target_to_string[c] for c in df["target"]]})


# In[ ]:


scatter_source = ColumnDataSource(data=scatter_df)


# ## PCA Transformed Plot

# In[ ]:


plot = figure()

plot.circle(x="x_comp_0", y="x_comp_1", source=scatter_source, color="color", legend="label")
show(plot)


# In[ ]:


def get_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    
    print(f'''
Confusion Matrix: 
{cm}
F1 Score: {f1}
Accuracy: {accuracy}
ROC AUC: {roc_auc}
    ''')
    
    return cm, f1, accuracy, roc_auc


# # Split Dataset

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.3, stratify=y)


# # Classify on PCA'd data (inconsistent results)

# In[ ]:


clf = GridSearchCV(estimator=SGDClassifier(), param_grid={"loss": ["log", "hinge"], "penalty": ["l1", "l2"], "alpha": [1, 1e-1, 1e-2], "max_iter": [10, 1000, ]}, cv=5)
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)

print(f"Test Metrics using {clf.estimator.__class__}")
test_cm, test_f1, test_acc, test_auc = get_metrics(y_test, y_predict)
print(clf.best_params_)

sgd_clf = clf


# In[ ]:


clf = GridSearchCV(estimator=SVC(), param_grid={"gamma": ["scale"], "C": [10], "kernel": ["linear"]}, cv=5)
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)

print(f"Test Metrics using {clf.estimator.__class__}")
test_cm, test_f1, test_acc, test_auc = get_metrics(y_test, y_predict)
print(clf.best_params_)

svc_clf = clf


# In[ ]:


clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [50, 75, 100, 300], "max_depth": [1, 10]}, cv=5).fit(x_train, y_train)#
y_predict = clf.predict(x_test)

print(f"Test Metrics using {clf.estimator.__class__}")
test_cm, test_f1, test_acc, test_auc = get_metrics(y_test, y_predict)
print(clf.best_params_)

rfc_clf = clf


# # Plot Decision Boundary

# In[ ]:


def plot_boundary(X, y, clfs, titles):
    
    color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA']#, '#AAAAFF']
    color_list_bold = ['#EEEE00', '#000000', '#00CC00']#, '#0000CC']
    custom_cmap2 = ListedColormap(color_list_light)
    custom_cmap1 = ListedColormap(color_list_bold)
    # Plotting decision regions
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    print(xx.shape)
    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

    for idx, clf, tt in zip(product([0, 1], [0, 1]),
                            clfs,
                            titles):

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.8,cmap=custom_cmap2)
        axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,cmap=custom_cmap1,
                                      s=20, edgecolor='black')
        axarr[idx[0], idx[1]].set_title(tt)
    plt.show()


# In[ ]:


plot_boundary(x_pca_orig[:, :dimension], y, [sgd_clf, rfc_clf, svc_clf], ["SGD", "Random Forest", "SVC"])


# # Clustering

# In[ ]:


def plot_cluster(x, dim_1, dim_2, n_clusters=None):
    
    best_silh_score, best_kmeans = -np.inf, None
    best_n_cluster = None
    if n_clusters is None:
        for n_clusters in [2, 3, 4, 5]:
            kmeans = KMeans(n_clusters=n_clusters)
            dim_slice = [dim_1, dim_2]

            kmeans.fit(x[:, dim_slice], y_train)
            y_predict = kmeans.predict(x[:, dim_slice])
            
            sil_score = silhouette_score(x[:, dim_slice], y_predict)
            
            print(f"Silhouette score for {n_clusters} clusters is {sil_score}")
            
            if sil_score > best_silh_score:
                best_silh_score = sil_score
                best_kmeans = kmeans
                best_n_cluster = n_clusters
    
    n_clusters = best_n_cluster
    kmeans = best_kmeans
    y_predict = kmeans.predict(x[:, dim_slice])
    
    clusters = kmeans.cluster_centers_
    colors = ["red", "blue", "yellow", "green", "black", "grey"]
    
    cluster_results_df = pd.DataFrame({"x_comp_0": x[:, dim_1], "x_comp_1": x[:, dim_2], "color": [colors[i] for i in y_predict]})
    cluster_results_source = ColumnDataSource(data=cluster_results_df)
    
    cluster_center_df = pd.DataFrame({"x_comp_0": clusters[:, 0], "x_comp_1": clusters[:, 1],
                                  "color": colors[:n_clusters],
                                  "label": [f"Cluster {i}" for i in range(len(clusters))]})
    cluster_source = ColumnDataSource(data=cluster_center_df)
    
    print(f"Plot of {list(df)[dim_1]} vs {list(df)[dim_2]}")
    
    plot = figure()

    plot.circle(x="x_comp_0", y="x_comp_1", source=cluster_results_source, color="color")
    plot.diamond(x="x_comp_0", y="x_comp_1", source=cluster_source, color="color", legend="label", size=20)

    show(plot)


# In[ ]:


plot_cluster(x_orig, 0, 3)
plot_cluster(x_orig, 0, 4)
plot_cluster(x_orig, 0, 7)
plot_cluster(x_orig, 0, 9)

plot_cluster(x_orig, 3, 4)
plot_cluster(x_orig, 3, 7)
plot_cluster(x_orig, 3, 9)

plot_cluster(x_orig, 4, 7)
plot_cluster(x_orig, 4, 9)


plot_cluster(x_orig, 7, 9)


# In[ ]:




