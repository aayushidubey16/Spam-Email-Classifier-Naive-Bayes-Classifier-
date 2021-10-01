from math import log
import pandas as pd
import os
import re
import numpy as np
from nltk.corpus import words
import string

def naive_bayes_classifier(message):
   message = re.sub('\W', ' ', message)
   message = message.lower().split()

   p_spam_given_message = log(P_spam)
   p_ham_given_message = log(P_ham)
 
   for word in message:
      if word in parameters_spam:
         p_spam_given_message += log(parameters_spam[word])

      if word in parameters_ham:
         p_ham_given_message += log(parameters_ham[word])

   p_spam_given_message = p_spam_given_message
   p_ham_given_message = p_ham_given_message
   if p_ham_given_message > p_spam_given_message:
      return 'ham'
   elif p_spam_given_message > p_ham_given_message:
      return 'spam'
   else:
      return 'needs human classification'

def read_content_from_text_files(mailpath):
    files = os.listdir(mailpath)
    exclude = set(string.punctuation)
    set_word = set(words.words())
    content = []
    for file in files:
        srcpath = os.path.join(mailpath, file)
        fp = open(srcpath, errors="ignore")
        data = fp.read()
        data = ' '.join([i for i in data.split() if i in set_word])
        data = ' '.join([a for a in data.split() if a not in exclude])
        label = int(file.split("_")[1].replace(".txt","")) 
        content.append([label,data])
    return pd.DataFrame(content, columns =['Id', 'Message'])

#Main Function Starts From Here
#Label CSV
dfLabel = pd.read_csv('emails\\labels.csv')
dfLabel['Label'] = np.where(dfLabel['Label'] == 0, 'spam', 'ham')
#Train Content DF 
dfContent = read_content_from_text_files('emails\\TR_CH')
#Label & Content merged DF
dfMerge = pd.merge(dfContent,dfLabel,on="Id")
dfMerge = dfMerge.drop("Id",1)
print(dfMerge)

#Spliting merged DF in train & test
data_randomized = dfMerge.sample(frac=1, random_state=1)
training_test_index = round(len(data_randomized) * 0.8)
training_set = data_randomized[:training_test_index].reset_index(drop=True)
testing_set = data_randomized[training_test_index:].reset_index(drop=True)

#building vocabulary
vocabulary = []
for sms in training_set.Message:
    sms = sms.split(' ')
    for word in sms:
        vocabulary.append(word)

vocabulary = list(set(vocabulary))

#building word count matrix
word_counts_per_sms = {unique_word: [0] * len(training_set.Message) for unique_word in vocabulary}

for index, sms in enumerate(training_set.Message):
    sms = sms.split(' ')
    for word in sms:
        word_counts_per_sms[word][index] += 1

word_counts = pd.DataFrame(word_counts_per_sms)
print('vocabulary length : '+str(len(vocabulary)))

#merging word count matrix with training set
training_set = pd.concat([training_set,word_counts],axis=1)

alpha = 1
# Isolating spam and ham messages
spam_messages = training_set[training_set['Label'] == 'spam']
ham_messages = training_set[training_set['Label'] == 'ham']

#Calculating probablities
P_spam = len(spam_messages) / len(training_set)
P_ham = len(ham_messages) / len(training_set)

# N_Spam total of word count of spam messeges
n_words_per_spam_message = spam_messages['Message'].apply(len)
n_spam = n_words_per_spam_message.sum()

# N_Ham total of word count of ham messeges
n_words_per_ham_message = ham_messages['Message'].apply(len)
n_ham = n_words_per_ham_message.sum()

# N_Vocabulary
n_vocabulary = len(vocabulary)

# Initiate parameters
parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}

# Calculate probality word wise for spam & ham messages to train classifier
for word in vocabulary:
   if word != 'Label' and word != 'Message':  
    n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
    p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
    parameters_spam[word] = p_word_given_spam

    n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
    p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
    parameters_ham[word] = p_word_given_ham

#testing classifier
testing_set['Predicted'] = testing_set.Message.apply(naive_bayes_classifier)
print(testing_set.Predicted.value_counts(normalize=True))
correct = 0
total = testing_set.shape[0]
for row in testing_set.iterrows():
   row = row[1]
   if row['Label'] == row['Predicted']:
      correct += 1
print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)

#Predicting Labels for given dataset
#Train Content DF 
dfContentDS = read_content_from_text_files('emails\\TT_CH')
dfContentDS['Prediction'] = dfContentDS.Message.apply(naive_bayes_classifier)
print(dfContentDS.Prediction.value_counts(normalize=True))
dfContentDS['Prediction'] = np.where(dfContentDS['Prediction'] == 'spam', 0, 1)
header = ["Id", "Prediction"]
dfContentDS.to_csv("predicted_output.csv", index=False, columns=header)