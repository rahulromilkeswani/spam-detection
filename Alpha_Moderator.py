import re
import random
import math
import numpy as np
import matplotlib.pyplot as plt 
import Spam_Classifier as sc

random.seed(10)
"""
Read text data from file and pre-process text by doing the following
1. convert to lowercase
2. convert tabs to spaces
3. remove "non-word" characters
Store resulting "words" into an array
"""
FILENAME='SMSSpamCollection'
all_data = open(FILENAME).readlines()
# split into train and test
num_samples = len(all_data)
all_idx = list(range(num_samples))
random.shuffle(all_idx)
idx_limit = int(0.8*num_samples)
train_idx = all_idx[:idx_limit]
test_idx = all_idx[idx_limit:]
train_examples = [all_data[ii] for ii in train_idx]
test_examples = [all_data[ii] for ii in test_idx]

# Preprocess train and test examples
train_words = []
train_labels = []
test_words = []
test_labels = []

# train examples
for line in train_examples:
    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carriage return
    line = line.lower()  # lowercase
    line = line.replace("\t", ' ')  # convert tabs to space
    line_words = re.findall(r'\w+', line)
    line_words = [xx for xx in line_words if xx != '']  # remove empty words

    label = line_words[0]
    label = 1 if label == 'spam' else 0
    line_words = line_words[1:]
    train_words.append(line_words)
    train_labels.append(label)
    
# test examples
for line in test_examples:
    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carriage return
    line = line.lower()  # lowercase
    line = line.replace("\t", ' ')  # convert tabs to space
    line_words = re.findall(r'\w+', line)
    line_words = [xx for xx in line_words if xx != '']  # remove empty words

    label = line_words[0]
    label = 1 if label == 'spam' else 0

    line_words = line_words[1:]
    test_words.append(line_words)
    test_labels.append(label)

spam_words = []
ham_words = []

for ii in range(len(train_words)):  # we pass through words in each (train) SMS
    words = train_words[ii]
    label = train_labels[ii]
    if label == 1:
        spam_words += words
    else:
        ham_words += words
input_words = spam_words + ham_words  # all words in the input vocabulary

# Count spam and ham occurances for each word
num_spam = len(spam_words)
num_ham = len(ham_words)

#print(spam_counts['free'] / (num_spam + alpha * 20000))  # prob of "free" | spam
#print(ham_counts['free'] / (num_ham + alpha * 20000))  # prob of "free" | ham

#Pobability of spam and ham messages in the dataset
total_spam_probability = num_spam/len(input_words)
total_ham_probability = num_ham/len(input_words)

#Calculating measures for test and training data with alpha = 0.1
print("Alpha Value = 0.1")
print("")
print("TEST DATA")
print(sc.spam_classifier(0.1,test_words,test_labels,num_spam,num_ham,total_spam_probability,total_ham_probability,spam_words,ham_words))
print("")
print("TRAINING DATA")
print(sc.spam_classifier(0.1,train_words,train_labels,num_spam,num_ham,total_spam_probability,total_ham_probability,spam_words,ham_words))
print('\n\n')

test_i_value=[]
train_i_value=[]
test_f_scores=[]
test_accuracies = []
train_f_scores=[]
train_accuracies=[]

#Calculating measures for different values of alpha. 
for i in range(-5,1):
    alpha = 2**i
    measures_list=sc.spam_classifier(alpha,test_words,test_labels,num_spam,num_ham,total_ham_probability,total_ham_probability,spam_words,ham_words)
    test_i_value.append(i)
    test_f_scores.append(measures_list['F-Score'])
    test_accuracies.append(measures_list['Accuracy'])
for i in range(-5,1):
    alpha = 2**i
    measures_list=sc.spam_classifier(alpha,train_words,train_labels,num_spam,num_ham,total_ham_probability,total_ham_probability,spam_words,ham_words)
    train_i_value.append(i)
    train_f_scores.append(measures_list['F-Score'])
    train_accuracies.append(measures_list['Accuracy'])

#plotting the measures to a graph
fig = plt.figure()
testplot = fig.add_subplot(1, 2, 1)
trainplot = fig.add_subplot(1, 2, 2)
testplot.plot(test_i_value,test_f_scores, label="F-Score")
testplot.plot(test_i_value,test_accuracies,label="Accuracy")
testplot.set_xlabel('i Values')
testplot.set_ylabel('F-Score/Accuracy')
testplot.set_title('MEASURES FOR TEST SET')
testplot.legend()
trainplot.plot(train_i_value,train_f_scores, label="F-Score")
trainplot.plot(train_i_value,train_accuracies,label="Accuracy")
trainplot.set_xlabel('i Values')
trainplot.set_ylabel('F-Score/Accuracy')
trainplot.set_title('MEASURES FOR TRAINING SET')
trainplot.legend()
plt.show()

