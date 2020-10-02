import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")



    numeric_values = train._get_numeric_data()

    remove_na = numeric_values.fillna(numeric_values.mean())

    remove_na.to_csv("datumdanny.csv")
    h2o.init()
    h2odata = h2o.import_file("datumdanny.csv")
    print(h2odata.columns)
    train,test,valid = h2odata.split_frame(ratios=(0.7,0.15), seed = 123456)
    print(train)
    print(test)
    print(valid)
    selColumns = h2odata.columns
    print(selColumns)
    selColumns.remove("SalePrice")
    print(selColumns)

    glm_classifier = H2OGeneralizedLinearEstimator(family="binomial", nfolds=10, alpha=0.5)
    glm_classifier.train(y="SalePrice", x=selColumns, training_frame=train)

    # Predict using the GLM model and the testing dataset
    #predict = glm_classifier.predict(test)

    # View a summary of the prediction
    #predict.head()