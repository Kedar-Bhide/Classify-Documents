import os
import math


def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset


def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r', encoding='utf-8') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])


def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    f = open(filepath, "r", encoding='utf-8')
    line_list = [line.strip() for line in f]
    if len(line_list) > 0:
        for line in line_list:
            if vocab.count(line) == 0:
                if None not in bow.keys():
                    bow[None] = 1
                else:
                    bow[None] += 1
                continue
            if line not in bow.keys():
                bow[line] = 1
            else:
                bow[line] += 1

    f.close()

    return bow


def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1  # smoothing factor
    logprob = {}
    total_docs = 0
    label_16 = 0
    label_20 = 0
    for data in training_data:
        if data['label'] == '2016':
            label_16 = label_16 + 1
        if data['label'] == '2020':
            label_20 = label_20 + 1
        total_docs = total_docs + 1

    logprob['2020'] = math.log((label_20 + smooth)) - math.log((total_docs + 2))
    logprob['2016'] = math.log((label_16 + smooth)) - math.log((total_docs + 2))

    return logprob


def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1  # smoothing factor
    word_prob = {}
    wordCount = 0
    none_count = 0

    for word in vocab:
        word_prob[word] = 0

    for dict in training_data:
        if dict['label'] == label:
            for val in dict['bow']:
                wordCount = wordCount + dict['bow'][val]
                if val == None:
                    none_count = none_count + dict['bow'][val]
                    continue
                word_prob[val] = word_prob[val] + dict['bow'][val]

    for word in vocab:
        word_prob[word] = math.log(word_prob[word] + smooth * 1) - math.log(wordCount + smooth * (len(vocab) + 1))
        word_prob[None] = math.log(none_count + smooth * 1) - math.log(wordCount + smooth * (len(vocab) + 1))

    return word_prob


def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    training_data = load_training_data(create_vocabulary(training_directory, cutoff), training_directory)
    vocab = create_vocabulary(training_directory, cutoff)

    retval['vocabulary'] = vocab
    retval['log prior'] = prior(training_data, label_list)
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, '2020')
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, training_data, '2016')

    return retval


def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    prior_2016 = model['log prior']['2016']
    prior_2020 = model['log prior']['2020']
    bow = create_bow(model['vocabulary'], filepath)
    log_p_2016 = 0
    log_p_2020 = 0

    for key in model['log p(w|y=2016)']:
        if key in bow:
            log_p_2016 += model['log p(w|y=2016)'][key] * bow[key]

    for key in model['log p(w|y=2020)']:
        if key in bow:
            log_p_2020 += model['log p(w|y=2020)'][key] * bow[key]

    retval['log p(y=2020|x)'] = prior_2020 + log_p_2020
    retval['log p(y=2016|x)'] = prior_2016 + log_p_2016


    if retval['log p(y=2016|x)'] > retval['log p(y=2020|x)']:
        retval['predicted y'] = '2016'
    else:
        retval['predicted y'] = '2020'

    return retval

print(create_vocabulary('./EasyFiles/', 1))