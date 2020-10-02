import glob
import keras as k
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("../input/kidney_disease.csv")

    # acc to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5497482/
    # retain only some columns
    columns_to_retain = ["sg", "al", "sc", "hemo",
                         "pcv", "wbcc", "rbcc", "htn", "classification"]
    #columns_to_retain = df.columns
    df = df.drop(
        [col for col in df.columns if not col in columns_to_retain], axis=1)
    # now drop the rows with na values
    df = df.dropna(axis=0)

    for column in df.columns:
        if df[column].dtype == np.number:
            continue
        df[column] = LabelEncoder().fit_transform(df[column])

    X = df.drop(["classification"], axis=1)
    y = df["classification"]

    x_scaler = MinMaxScaler()
    x_scaler.fit(X)
    column_names = X.columns
    X[column_names] = x_scaler.transform(X)

    X_train,  X_test, y_train, y_test = train_test_split(
        X, y, test_size=5, shuffle=True)

    optimizer = k.optimizers.Adam()
    checkpoint = ModelCheckpoint("ckd.best.model", monitor="loss",
                                 mode="min", save_best_only=True, verbose=0)

    model = k.models.Sequential()
    model.add(Dense(256, input_dim=len(X.columns),
                    kernel_initializer=k.initializers.random_normal(seed=13), activation="relu"))
    model.add(Dense(1, activation="hard_sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=2000, batch_size=X_train.shape[0],
              callbacks=[checkpoint], verbose=0)
    model.save("ckd.model")
    plt.plot(history.history["acc"])
    plt.plot(history.history["loss"])
    plt.title("model accuracy & loss")
    plt.ylabel("accuracy and loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.grid(which="both")
    plt.savefig("ckd.png")

    print("*******************************************************************")
    print("Shape of training data: {0}".format(X_train.shape))
    print("Shape of test data    : {0}".format(X_test.shape))
    print("*******************************************************************")

    for model_file in glob.glob("*.model"):
        print("Model file: {0}".format(model_file))
        model = k.models.load_model(model_file)
        pred = model.predict(X_test)
        scores = model.evaluate(X_test, y_test)
        print("Original  : {0}".format(", ".join([str(x) for x in y_test])))
        print("Predicted : {0}".format(", ".join([str(x) for x in pred])))
        print("Scores    : {0}".format(", ".join(["{0} = {1}".format(model.metrics_names[i], scores[i]) for i in range(len(model.metrics_names))])))
        print("*******************************************************************")
    pass


if __name__ == '__main__':
    main()
