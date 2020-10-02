import h2o
import h2o.frame
import numpy as np
from tqdm import tqdm
import h2o.model.metrics_base
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator

global train_data
global test_data


def main():

    global train_data
    global test_data

    complete_dataset = "../input/creditcard.csv"

    # Start H2O on your local machine
    h2o.init()

    full_data = h2o.import_file(complete_dataset)

    (train_data, test_data) = full_data.split_frame([0.7])

    model_build(12)


def model_build(i):

    #
    # Train deep autoencoder learning model on "normal"
    # training data, y ignored
    #
    anomaly_model = H2OAutoEncoderEstimator(
        activation="Tanh",
        hidden=[i],
        sparse=True,
        l1=1e-4,
        epochs=10,
        ignored_columns=[train_data.names[0],train_data.names[train_data.ncol-1]]

    )

    anomaly_model.train(x=train_data.names, training_frame=train_data)


    # Compute reconstruction, error with the Anomaly
    # detection app (MSE between output and input layers)
    recon_error = anomaly_model.anomaly(test_data, False)
    error_str = recon_error.get_frame_data()

    err_list = list(map(float, error_str.split("\n")[1:-1]))
    mse_error = anomaly_model.mse()
    multi = 8

    threshold = mse_error*multi
    # Define a threshold to select outliers 
    
    # print "The following test points are reconstructed with an error greater than: ", threshold

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    data_str = test_data.get_frame_data()
    lbl_list = data_str.split("\n")


    for i in tqdm(range(len(recon_error) - 1)):
        if err_list[i] > threshold:
            if lbl_list[i+1].split(",")[-1] == "0":
                fp += 1
            else:
                tp += 1
        else:
            if lbl_list[i+1].split(",")[-1] == "0":
                tn += 1
            else:
                fn += 1
    recall =0

   
    print("TP :", tp, "/n")
    print("FP :", fp, "/n")
    print("TN :", tn, "/n")
    print("FN :", fn, "/n")

    if tp+fp != 0:

        recall = (100*float(tp))/(tp+fn)
        print("Recall (sensitivity) true positive rate (TP / (TP + FN)) :", recall)
        print("Precision (TP / (TP + FP) :", (100*float(tp))/(tp+fp))
        print("F1 score (harmonic mean of precision and recall (sensitivity)) (2TP / (2TP + FP + FN)) :", 200*float(tp)/(2*tp+fp+fn))
        print("Accuracy (TP+TN)/TestDataSetSize: ", float(tp + tn)  / (test_data.nrow))

    return recall


if __name__ == '__main__':
    main()


