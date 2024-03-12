import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Need to convert the values in one-hot encoding
enc = OneHotEncoder()
possible_labels = np.array([0, 1, 2]).reshape(-1, 1)
enc.fit(possible_labels)

def calculate_metrics(predictions, target, mode = None):
    # AUC calculation
    auc, _, _ = roc_auc(predictions, target)

    # Precision, Recall, and F1 calculation
    mcc, f1, precision, recall = scores_custom(predictions, target)

    print(f'MCC={mcc:.3f} | F1-score={f1:.3f} | Precision={precision:.3f} | Recall={recall:.3f} | AUC={auc:.3f}')

    return mcc, f1, precision, recall, auc


def roc_auc(predictions, target):
    # Converting raw scores into probabilities
    specificity, sensitivity = find_vals(predictions, target)
    predictions = torch.softmax(predictions, dim=1)
    predictions, target = predictions.cpu().numpy(), target.cpu().numpy()
    target_one_hot = enc.transform(target.reshape(-1, 1)).toarray()  # Reshaping needed by the library

    # Arguments take 'GT' before taking 'predictions'
    return roc_auc_score(target_one_hot, predictions), specificity, sensitivity

def scores_custom(predictions, target, average='micro'):
    predictions, precision, recall = find_vals_multi(predictions, target, average)
    predictions, target = predictions.cpu().numpy(), target.cpu().numpy()
    
    #MCC
    mcc = matthews_corrcoef(target, predictions)
    
    # F1 score
    f1 = f1_score(target, predictions, average=average)
    
    return mcc, f1, precision, recall

def find_vals_multi(predictions, target, average):
    predictions = torch.max(predictions, dim=1)[1]  # We need the indices for the max
    print(predictions)
    print(target)
    
    cm = confusion_matrix(predictions.cpu().numpy(), target.cpu().numpy())
    print(cm)
    
    # Calculate precision, recall, and F1 score for each class
    precision = precision_score(target.cpu().numpy(), predictions.cpu().numpy(), average=None)
    recall = recall_score(target.cpu().numpy(), predictions.cpu().numpy(), average=None)

    # Calculate average precision, recall, and F1 score
    precision_avg = precision_score(target.cpu().numpy(), predictions.cpu().numpy(), average=average)
    recall_avg = recall_score(target.cpu().numpy(), predictions.cpu().numpy(), average=average)

    return predictions, precision_avg, recall_avg

def acc_pred(predictions, target):
    # Converting raw scores into probabilities
    preds = torch.max(predictions, dim=1)[1]
    correct = (preds == target).sum()
    return correct/target.size(0)



def find_vals(predictions, target):
    predictions = torch.max(predictions, dim=1)[1]  # We need the indices for the max
    print(predictions)
    print(target)
    cm = confusion_matrix(predictions.cpu().numpy(), target.cpu().numpy())
    print(cm)
    specificity = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    print('Combined:')
    print('specificity:', specificity)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    print('sensitivity:', sensitivity)
    return specificity, sensitivity


if __name__ == '__main__':
    y = np.array([0, 1, 1, 0])
    # y = np.array([0, 1, 1, 0]).reshape(-1, 1)
    # enc.fit(possible_labels)
    # y = enc.transform(y).toarray()
    # print(y)
    # x = torch.randn((4, 2))
    x = torch.as_tensor([
        [0.9, 0.1],
        [0.9, 2.1],
        [0.9, 2.1],
        [0.9, 0.1]
    ])
    # y = torch.as_tensor(y).squeeze()
    y = torch.as_tensor(y)
    print(acc_pred(x, y))
