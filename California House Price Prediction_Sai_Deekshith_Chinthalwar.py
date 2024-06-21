
# Commented out IPython magic to ensure Python compatibility.
# %pylab inline
import pandas as pd
from scipy import linalg
from sklearn import tree
from itertools import combinations
import scipy
import scipy.io as io
from scipy.io import mmread
import scipy.sparse as sparse

"""

---



---



**Data access:**

The training data is stored in https://drive.google.com/file/d/1WfFkiKLBzTRh8zGDNYQXGUnq8-DK757m/view?usp=sharing.

The testing data is stored in https://drive.google.com/file/d/1Met2KysUV0shr6t2JiS7RWXMAKiueiG2/view?usp=sharing.


Once you open the link in the brower, make sure you click the "Add shortcut to Drive" and now your google drive should show up the two csv files.  Then you run the following code to link colab to your google drive.

**Data description:**
The task is to predict house sale prices based on the house information, such as # of bedrooms, living areas, locations, near-by schools, and the seller summary. It covers almost every aspects of residential homes. The data consist of houses sold in California on 2020, with 30000 training labeled dataset and 15000 unlabeled dataset.


**Metric/Score:** Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)


"""

from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir("/content/drive/My Drive")

df_train = pd.read_csv('house_train.csv')
df_test  = pd.read_csv('house_test.csv')

df_train

df_train.iloc[:, 20:35]

df_train.info()

df_train.columns

df_train.describe()

df_test

df_test.columns

df_test.info()

# Your code starts here
numerical_data = df_train.select_dtypes("number")
numerical_data.hist(bins=20, figsize=(12, 22), edgecolor="black",
                    layout=(9, 4))
plt.subplots_adjust(hspace=0.8, wspace=0.8)

numerical_data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


correlation_matrix = numerical_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

housing = df_train.drop("Sold Price", axis=1)
housing_labels = df_train["Sold Price"].copy()

df_full=pd.concat([housing,df_test],ignore_index= True)

df_full['Listed On'] = pd.to_datetime(df_full['Listed On'])

df_full['Listed Year'] = df_full['Listed On'].dt.year
df_full['Year built'].fillna(mean)
df_full['age']=df_full['Listed Year']-df_full['Year built']
df_full['Total School Distance']=df_full['Elementary School Distance']+df_full['Middle School Distance']+df_full['High School Distance']
df_full['Total School Score']=df_full['Elementary School Score']+df_full['Middle School Score']+df_full['High School Score']
df_full['Total Area']=df_full['Lot'] + df_full['Total interior livable area'] + df_full['Total spaces']
df_full['Bathroom Ratio'] = (df_full['Full bathrooms'] + 1)/(df_full['Bathrooms'] + 1 )
df_full

housing=df_full[:30000]
df_test=df_full[30000:]

#pd.plotting.scatter_matrix(numerical_data, figsize=(12, 8))
#plt.show()

df_train.plot(kind="scatter", y="Total interior livable area", x="Sold Price", grid=True,
             s=df_train["Bathrooms"], label="Bathrooms",
             c="Total spaces", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.show()

df_train.plot(kind="scatter", x="Sold Price", y="Tax assessed value",
             alpha=0.5, grid=True)
plt.show()

from sklearn.metrics.pairwise import rbf_kernel
ages = np.linspace(housing["Total School Score"].min(),
                   housing["Total School Score"].max(),
                   500).reshape(-1, 1)
gamma1 = 0.1
gamma2 = 0.03

age_simil_35 = rbf_kernel(ages, [[20]], gamma=gamma1)
rbf2 = rbf_kernel(ages, [[20]], gamma=gamma2)

fig, ax1 = plt.subplots()

ax1.set_xlabel("Total School Score")
ax1.hist(housing["Total School Score"], bins=50)

ax2 = ax1.twinx()  # create a twin axis that shares the same x-axis
color = "blue"
ax2.plot(ages, age_simil_35, color=color, label="gamma = 0.10")
ax2.plot(ages, rbf2, color=color, label="gamma = 0.03", linestyle="--")
ax2.tick_params(axis='y', labelcolor=color)

plt.legend(loc="upper left")
plt.show()

housing['Last Sold Price']=housing['Last Sold Price']+1
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["Last Sold Price"].hist(ax=axs[0], bins=50)
housing["Last Sold Price"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Last Sold Price")
axs[1].set_xlabel("Log of Last Sold Price")
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["Annual tax amount"].hist(ax=axs[0], bins=50)
housing["Annual tax amount"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Annual tax amount")
axs[1].set_xlabel("Log of Annual tax amount")
plt.show()

housing['Elementary School Distance']=housing['Elementary School Distance']+1
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["Elementary School Distance"].hist(ax=axs[0], bins=50)
housing["Elementary School Distance"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Elementary School Distance")
axs[1].set_xlabel("Log of Elementary School Distance")
plt.show()

housing['High School Distance']=housing['High School Distance']+1
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["High School Distance"].hist(ax=axs[0], bins=50)
housing["High School Distance"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("High School Distance")
axs[1].set_xlabel("Log of High School Distance")
plt.show()

housing['Tax assessed value']=housing['Tax assessed value']+1
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["Tax assessed value"].hist(ax=axs[0], bins=50)
housing["Tax assessed value"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Tax assessed value")
axs[1].set_xlabel("Log of Tax assessed value")
plt.show()

housing['Middle School Distance']=housing['Middle School Distance']+1
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["Middle School Distance"].hist(ax=axs[0], bins=50)
housing["Middle School Distance"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Middle School Distance")
axs[1].set_xlabel("Log of Middle School Distance")
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["Annual tax amount"].hist(ax=axs[0], bins=50)
housing["Annual tax amount"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Annual tax amount")
axs[1].set_xlabel("Log of Annual tax amount")
plt.show()

housing['Listed Price']=housing['Listed Price']+1
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["Listed Price"].hist(ax=axs[0], bins=50)
housing["Listed Price"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Listed Price")
axs[1].set_xlabel("Log of Listed Price")
plt.show()

housing['Bathrooms']=housing['Bathrooms']+1
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["Bathrooms"].hist(ax=axs[0], bins=50)
housing["Bathrooms"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Bathrooms")
axs[1].set_xlabel("Log of Bathrooms")
axs[0].set_ylabel("Tax assessed value")
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["Total School Distance"].hist(ax=axs[0], bins=50)
housing["Total School Distance"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Total School Distance")
axs[1].set_xlabel("Log of Total School Distance")
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["Total Area"].hist(ax=axs[0], bins=50)
housing["Total Area"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Total Area")
axs[1].set_xlabel("Log of Total Area")
plt.show()

null_rows_idx = housing.isnull().any(axis=1)
housing.loc[null_rows_idx].head()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
housing_num = housing.select_dtypes("number")
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.mean().values)

X = imputer.transform(housing_num)
imputer.feature_names_in_
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)
housing_tr.loc[null_rows_idx].head()

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

from sklearn.preprocessing import StandardScaler
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(housing[["Listed Price"]], scaled_labels)
some_new_data = housing[["Listed Price"]].iloc[:5] # pretend this is new data
scaled_predictions = model.predict(some_new_data)

predictions = target_scaler.inverse_transform(scaled_predictions)

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
import pandas as pd
import numpy as np

housing.columns

from seaborn._stats.counting import DataFrame

default_num_pipeline = make_pipeline(SimpleImputer(strategy="mean"),
                                    StandardScaler())

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))


preprocessing = ColumnTransformer([
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ("default", default_num_pipeline,["Total School Score"])
    ])

housing_prepared = preprocessing.fit_transform(df_full)
housing_prepared.shape

housing=housing_prepared[:30000]
df_test=housing_prepared[30000:]

housing.shape

preprocessing.get_feature_names_out

"""#Linear Regression:"""

lr = LinearRegression()

lr.fit(housing, housing_labels.values)

housing_predictions2 = lr.predict(housing)
housing_predictions2.round(-2)

housing_labels.values - 1

error_ratios = housing_predictions2.round(-2) / housing_labels.values - 1
print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))

from sklearn.metrics import mean_squared_error

lin_rmse = mean_squared_error(housing_labels, housing_predictions2,
                              squared=False)
lin_rmse

from sklearn.model_selection import cross_val_score

lin_rmses = -cross_val_score(lr, housing, housing_labels.values,
                              scoring="neg_root_mean_squared_error", cv=10)

pd.Series(lin_rmses).describe()

"""#Decision Tree Regression:"""

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing, housing_labels.values)

housing_predictions3 = tree_reg.predict(housing)
housing_predictions3.round(-2)

housing_labels.values - 1

error_ratios = housing_predictions3.round(-2) / housing_labels.values - 1
print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))

from sklearn.metrics import mean_squared_error

tree_rmse = mean_squared_error(housing_labels, housing_predictions3,
                              squared=False)
tree_rmse

tree_rmse = -cross_val_score(tree_reg, housing, housing_labels.values,
                              scoring="neg_root_mean_squared_error", cv=10)

pd.Series(tree_rmse).describe()

"""#Random Forest Regression:"""

from sklearn.ensemble import RandomForestRegressor

forest_reg_pipeline = RandomForestRegressor()

forest_reg_pipeline.fit(housing[:1000], housing_labels[:1000])

housing_predictions4 = forest_reg_pipeline.predict(housing[:1000])
housing_predictions4.round(-2)

housing_labels[:1000].values - 1

error_ratios = housing_predictions4.round(-2) / housing_labels[:1000].values - 1
print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))

from sklearn.metrics import mean_squared_error

forest_rmse = mean_squared_error(housing_labels[:200], housing_predictions4[:200],
                              squared=False)
forest_rmse

forest_rmse = -cross_val_score(forest_reg_pipeline, housing[:200], housing_labels.values[:200],
                                scoring="neg_root_mean_squared_error", cv=10)

pd.Series(forest_rmse).describe()

"""#Tuning:"""

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

full_pipeline = Pipeline([
    ("random_forest", RandomForestRegressor(random_state=42)),
])

param_distribs = {'random_forest__max_features': randint(low=2, high=200)}

rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42)

rnd_search.fit(housing[:1000], housing_labels[:1000])

rnd_search.best_params_

cv_res = pd.DataFrame(rnd_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res = cv_res[["param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
score_cols = ["split0_test_score", "split1_test_score", "split2_test_score", "mean_test_score"]
cv_res.columns = [ "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)
cv_res.head()

final_model2 = rnd_search.best_estimator_  # includes preprocessing
feature_importances2 = final_model2["random_forest"].feature_importances_

feature_importances2.round(3)

np.savetxt("predictions_housing.txt", rnd_search.predict(df_test), delimiter=',')

rnd_search.predict(df_test).size

"""# Your Solution:
In the quest to find the best machine learning classifier for the housing dataset, I employed a RandomForestRegressor within a comprehensive pipeline. The preprocessing steps involved both numerical and categorical features, using a ColumnTransformer to apply specific transformations to each type. The RandomForestRegressor was chosen for its ability to capture complex relationships within the data because of its least crossvalidation means squared error value. To identify optimal hyperparameter settings, a GridSearchCV was employed with varying values for the 'max_features' parameter. This process involved cross-validating the model using a 3-fold cross-validation strategy to ensure robust performance assessment. The best-performing configuration was found to be when 'max_features' was set to 190. This classifier demonstrated superior predictive capabilities, striking a balance between capturing informative features and avoiding overfitting. Overall, the RandomForestRegressor, with carefully tuned hyperparameters and thorough preprocessing, emerged as the top-performing model for predicting housing prices in this scenario.
"""
