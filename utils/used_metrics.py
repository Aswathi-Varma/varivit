import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Need to convert the values in one-hot encoding
enc = OneHotEncoder()
possible_labels = np.array([0, 1]).reshape(-1, 1)
enc.fit(possible_labels)

def get_scores_bin(predictions, target):
    # Converting raw scores into probabilities
    specificity, sensitivity, f1_score, probabilities = find_values_bin(predictions, target)
    probabilities, target = probabilities.cpu().numpy(), target.cpu().numpy()
    return roc_auc_score(target, probabilities), specificity, sensitivity, f1_score

def get_scores_kfold(predictions, target):
    # Converting raw scores into probabilities
    specificity, sensitivity, f1_score, mcc = find_values(predictions, target)
    predictions = torch.softmax(predictions, dim=1)
    predictions, target = predictions.cpu().numpy(), target.cpu().numpy()
    probabilities = predictions[:, 1]
    target_one_hot = enc.transform(target.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # Arguments take 'GT' before taking 'predictions'
    return roc_auc_score(target_one_hot, predictions), specificity, sensitivity, f1_score, probabilities, mcc

def get_scores(predictions, target):
    # Converting raw scores into probabilities
    specificity, sensitivity, f1_score = find_values(predictions, target)
    predictions = torch.softmax(predictions, dim=1)
    predictions, target = predictions.cpu().numpy(), target.cpu().numpy()
    target_one_hot = enc.transform(target.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # Arguments take 'GT' before taking 'predictions'
    return roc_auc_score(target_one_hot, predictions), specificity, sensitivity, f1_score

def roc_auc(predictions, target):
    # Converting raw scores into probabilities
    specificity, sensitivity = find_vals(predictions, target)
    predictions = torch.softmax(predictions, dim=1)
    predictions, target = predictions.cpu().numpy(), target.cpu().numpy()
    target_one_hot = enc.transform(target.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # Arguments take 'GT' before taking 'predictions'
    return roc_auc_score(target_one_hot, predictions), specificity, sensitivity


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

def find_values_bin(predictions, target, threshold=0.5):
    probabilities = torch.sigmoid(predictions)
    binary_predictions = (probabilities > threshold).int()
    print(binary_predictions)
    print(target)
    
    cm = confusion_matrix(binary_predictions.cpu().numpy(), target.cpu().numpy())
    print(cm)
    
    specificity = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    print('Combined:')
    print('specificity:', specificity)
    
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    print('sensitivity:', sensitivity)
    
    f1 = f1_score(target.cpu().numpy(), binary_predictions.cpu().numpy())
    print('f1 score:', f1)
    
    return specificity, sensitivity, f1, probabilities

def find_values(predictions, target):
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

    #MCC
    mcc = matthews_corrcoef(target.cpu().numpy(), predictions.cpu().numpy())
    print('MCC:', mcc)

    return specificity, sensitivity, f1_score(target.cpu().numpy(), predictions.cpu().numpy()), mcc

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
