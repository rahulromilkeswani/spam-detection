
import re
import random
import math
import numpy as np

def spam_classifier(alpha_passed,message_words,message_labels,num_spam,num_ham,total_spam_probability,total_ham_probability,spam_words,ham_words):
    alpha = alpha_passed

    spam_counts = {}; ham_counts = {}
    # Spamcounts
    for word in spam_words:
        try:
            word_spam_count = spam_counts.get(word)
            spam_counts[word] = word_spam_count + 1
        except:
            spam_counts[word] = 1 + alpha  # smoothening
    for word in ham_words:
        try:
            word_ham_count = ham_counts.get(word)
            ham_counts[word] = word_ham_count + 1
        except:
            ham_counts[word] = 1 + alpha  # smoothening

    #List to store the predicted labels
    predicted_label=[]

    for message in message_words: #each message in test set
        message_spam_probability = message_ham_probability = 1 #initial spam and ham probability for the message
        for word in message : #each word in the message
            try:
                word_spam_probability = spam_counts[word]/float(num_spam+alpha*20000) #spam probability of the word based on count in the spam_counts dictionary
            except:
                word_spam_probability = alpha/float(num_spam+alpha*20000) #a small probability if word isn't present in the spam_counts dictionary
            message_spam_probability = message_spam_probability*word_spam_probability #mutiplying individual probability of words to calculating ham probability of the message

            try:
                word_ham_probability = ham_counts[word]/float(num_ham+alpha*20000)  #ham probability of the word based on count in the ham_counts dictionary
            except:
                word_ham_probability = alpha/float(num_ham+alpha*20000) #a small probability if word isn't present in the spam_counts dictionary
            message_ham_probability = message_ham_probability*word_ham_probability #mutiplying individual probability of words to calculate ham probability of the message

        #Adding the predicted labels to the list
        if(message_spam_probability*total_spam_probability>message_ham_probability*total_ham_probability):
            predicted_label.append(1)
        else:
            predicted_label.append(0)

    true_positive=false_positive=true_negative=false_negative=0
    score = 0

    #computing the confusion matrix
    for i in range(0,len(predicted_label)):
        if(message_labels[i]==1 and predicted_label[i]==1):
            true_positive+=1
            score+=1
        if(message_labels[i]==0 and predicted_label[i]==0):
           true_negative+=1
           score+=1
        if(message_labels[i]==0 and predicted_label[i]==1):
           false_positive+=1
        if(message_labels[i]==1 and predicted_label[i]==0):
           false_negative+=1
      

    #computing measures
    precision=true_positive/float(true_positive+false_positive)
    recall = true_positive/float(true_positive+false_negative)
    f_score=2*precision*recall/float(precision+recall)
    accuracy = score/len(message_labels)
    
    return_dict = { "True Positive": true_positive,
                   "False Positive" : false_positive,
                   "True Negative" : true_negative,
                   "False Negative": false_negative,
                   "Precision":precision,
                   "Recall":recall,
                   "F-Score":f_score,
                   "Accuracy":accuracy
                   }

    return return_dict
            


