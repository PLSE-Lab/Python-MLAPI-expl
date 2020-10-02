# this script consists of SMOTE method and LR. 30% data is used for testing.
# we get 0.79 AUPRC score.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score,average_precision_score, roc_curve, recall_score
import itertools
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)


class ImbalanceDataLR(object):
    def __init__(self, method):
        data = pd.read_csv('../input/creditcard.csv')
        # standardscaler the 'Amount' in data
        data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
        # do not depend on Time, so delete it
        data = data.drop(['Time', 'Amount'], axis=1)
        # features : X , Class : y
        X = data.ix[:, data.columns != 'Class']
        y = data.ix[:, data.columns == 'Class']
        # 147 fraud transanctions in y_test

        self.classes_name = [-1,1]
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        if method == "original":
            self.method = method
            self.X_train = X_train
            self.y_train = y_train
        elif method == "SMOTE":
            self.method = method
            sm = SMOTE(kind='regular',random_state=42)
            X_train, y_train = sm.fit_sample(X_train, y_train)
            self.X_train = pd.DataFrame(X_train)
            self.y_train = pd.DataFrame(y_train)
        elif method == "Random":
            self.method = method
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_sample(X_train, y_train)
            self.X_train = pd.DataFrame(X_train)
            self.y_train = pd.DataFrame(y_train)

    # select the best C parameters by Kfold method
    def printing_Kfold_scores(self):
        x_train_data = self.X_train
        y_train_data = self.y_train
        fold = KFold(5, True, random_state= 123)
        fold.get_n_splits(x_train_data)

        # Different C parameters
        c_param_range = [0.01, 0.1, 1, 10, 100]
        results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
        results_table['C_parameter'] = c_param_range

        # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
        j = 0
        for c_param in c_param_range:
            print('----------------------------------------')
            print('C parameter:', c_param)
            print('----------------------------------------')
            print('')

            recall_accs = []
            for iteration, indices in enumerate(fold.split(x_train_data), start=1):
              
                lr = LogisticRegression(C=c_param, penalty='l1')
                # Use the training data to fit the model. In this case, we use the portion of the fold to train the
                # model with indices[0].
                # We then predict on the portion assigned as the 'test cross validation' with indices[1]
                lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())
                # Predict values using the test indices in the training data
                y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

                # to a list for recall scores representing the current c_parameter
                recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
                recall_accs.append(recall_acc)
                print ('Iteration', iteration, ':recall score = ', recall_acc)
            # The mean value of those recall scores is the metric we want to save and get hold of.
            results_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
            j += 1
            print ('')
            print ('Mean recall score', np.mean(recall_accs))
            print('')

        best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']

        # Finally, we can check which C parameter is the best amongst the chosen.
        print ('*****************************************************************************')
        print ('Best model to choose from cross validation is the C parameter=', best_c)
        print ('*****************************************************************************')
        self.best_c = best_c
        # return best_c

    # compute the confusion matrix
    def plot_confusion_matrix(self, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting 'normalize=True'.
        """
        cm = self.cm
        classes = self.classes_name
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            #  print ("Normalized confusion matrix")
        else:
            1  # print('Confusion matrix, without normalization')

        # print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i,j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def main(self):
        # Use this C_parameter to build the final model with the whole training dataset and predict the classes
        # in the test dataset
        X_train = self.X_train
        y_train = self.y_train
        best_c  = self.best_c
        X_test  = self.X_test
        y_test  = self.y_test

        lr= LogisticRegression(C = best_c, penalty='l1')
        lr.fit(X_train,y_train.values.ravel())
        y_pred_undersample_proba = lr.predict_proba(X_test.values)
        # self.ROC_AUC = roc_auc_score(np.array(y_test)[:,0], y_pred_undersample_proba[:, 1])
        self.ROCPRC  = average_precision_score(np.array(y_test)[:,0], y_pred_undersample_proba[:, 1])

        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9]
        plt.figure(figsize=(10, 10))

        j=1
        for i in thresholds:
            y_test_predictions_high_recall = y_pred_undersample_proba[:,1] > i

            plt.subplot(3,3,j)
            j +=1

            # Compute confusion matrix
            cnf_matrix = confusion_matrix(y_test, y_test_predictions_high_recall)
            self.cm = cnf_matrix
            np.set_printoptions(precision=2)

            print("Recall metric in the testing dataset: ", float(cnf_matrix[1,1])/(cnf_matrix[1,0]+cnf_matrix[1,1]))

            # Plot non-normalized confusion matrix
            self.plot_confusion_matrix(title='Threshold = %s'%i)
        plt.savefig("Different thresholds for p_pred_undersample_proba_%s.png" % self.method, dpi =500)
        plt.close()

        # from itertools import cycle

        # colors = cycle(['navy','turquoise','darkorange','cornflowerblue','teal','red','yellow','green','blue','black'])
        plt.figure(figsize=(5, 5))

        self.precision, self.recall, _ = precision_recall_curve(y_test, y_pred_undersample_proba[:,1])
        plt.plot(self.recall, self.precision)
        # Plot Precisin-Recall curve
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall')
        plt.legend(loc="lower left", fontsize=8)
        plt.savefig("Precision-Recall_%s.png" % self.method, dpi= 200)
        plt.figure(figsize=(5, 5))

class InitialAnalysis(object):
    def __init__(self):
        self.data = pd.read_csv('../input/creditcard.csv')
    def plot1(self):
      # plot comparision between fraud and Normal

        count_classes = pd.value_counts(self.data['Class'], sort=True).sort_index()
        df=pd.Series(np.array([284315,492]),index=['-1','+1'])
        df.plot(kind='bar')
        # plt.title("Fraud class histogram")
        plt.xlabel('class')
        plt.ylabel('amount')
        plt.savefig('Fraud class histogram.png', dpi= 200)
        plt.close()
if __name__ == "__main__":
    p=InitialAnalysis()
    p.plot1()

    # data = ImbalanceDataLR("original")
    # data = ImbalanceDataLR("Random")
    data = ImbalanceDataLR("SMOTE")
    data.printing_Kfold_scores()

    data.main()
    print(data.method, data.best_c, data.ROCPRC)