import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import sklearn


class Trainer:

    def __init__(self, device, model, optimizer, weight=None):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.weight = weight
        self.epoch = 0

    def train(self, train_loader, print_interval=10):
        self.epoch += 1
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data).to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.weight is not None:
                loss = F.nll_loss(output, target, weight=self.weight)
            else:
                loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if print_interval is not None:
                if batch_idx % print_interval == 0:
                    print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        self.epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

    def validate(self, validate_loader, name="VALIDATION"):
        self.model.eval()
        validate_loss = 0
        correct = 0
        f1_scores = []
        predictions = np.array([-1, -1])  # Necessary to allow stacking
        with torch.no_grad():
            for data, target in validate_loader:
                data, target = Variable(data).to(self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss and other scores
                validate_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                f1_scores.append(sklearn.metrics.f1_score(target.cpu().numpy(), pred.cpu().numpy(),average="macro"))
                for ind_pred, ind_target in zip(pred, target):
                    labeled_pred = (ind_pred.cpu().numpy(), ind_target.cpu().numpy())
                    predictions = np.vstack((predictions, labeled_pred))
        validate_loss /= len(validate_loader.dataset)
        f1_score = sum(f1_scores) / len(f1_scores)

        print(name+": Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), F1:{:.4f}".format(
            validate_loss, correct, len(validate_loader.dataset),
            100. * correct / len(validate_loader.dataset),
            f1_score))
        return predictions[1:].astype(int)

    def predict_test(self, test_loader, label_classes):
        self.model.eval()
        predictions = np.array([None, None])  # Necessary to allow stacking
        index = 0
        with torch.no_grad():
            for data in test_loader:
                data = Variable(data).to(self.device)
                output = self.model(data)
                pred = output.max(1, keepdim=True)[1]
                for individual in pred:
                    pred_name = label_classes[individual]
                    predictions = np.vstack((predictions, (index, pred_name)))
                    index += 1
        return predictions[1:]  # Everything after the junk 0th entry

    def predicted_histogram_stats(self, labeled_predictions, label_classes):
        true_labels = labeled_predictions[:,1]
        true_bins = np.bincount(np.array(true_labels), minlength=len(label_classes))

        predictions = labeled_predictions[:,0]
        pred_bins = np.bincount(predictions.astype(int), minlength=len(label_classes))

        print("Class \t # Real Occurance \t # Pred Occurance")
        for idx, name in enumerate(label_classes):
            print(name+":\t\t", true_bins[idx], "\t\t\t", pred_bins[idx])
            
    def confusion_matrix(self, labeled_predictions):
        confusion_matrix = sklearn.metrics.confusion_matrix(
                            labeled_predictions[:,1], labeled_predictions[:,0])
        plt.imshow(confusion_matrix)
        plt.colorbar()
        plt.show()
        return confusion_matrix
