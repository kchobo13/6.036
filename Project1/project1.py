from string import punctuation, digits

import numpy as np
import matplotlib.pyplot as plt

### Parameters
T_perceptron = 5

T_avperceptron = 5

T_avgpa = 5
L_avgpa = 10
###

test_feature_matrix = np.array([[1,0,0],[0,1,1],[1,0,1],[0,1,0]])
test_labels = np.array([-1,1,-1,1])
test_theta = np.array([1,1,1])
test_theta_0 = 1
test_feature_vector = test_feature_matrix[0]
test_label = test_labels[0]

### Part I

def hinge_loss(feature_matrix, labels, theta, theta_0):
    """
    Section 1.2
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """

    avg_hinge = 0.0
    number = len(feature_matrix)
    for index in xrange(number):
        point = feature_matrix[index]
        label = labels[index]

        hinge = float((np.dot(point, theta) + theta_0) * label)
        if hinge <= 1:
            avg_hinge += 1-hinge

    avg_hinge *= float(1.0/number)
    return avg_hinge

def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Section 1.3
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """

    
    if (1-hinge_loss([feature_vector], [label], current_theta, current_theta_0) <= 0):
        current_theta += label*feature_vector
        current_theta_0 += label
    
    return (current_theta, current_theta_0)

def perceptron(feature_matrix, labels, T):
    """
    Section 1.4
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """

    number = len(feature_matrix)
    total = feature_matrix.size
    theta = np.zeros(total/number)
    theta_0 = 0.0

    for iteration in xrange(T):
        for index in xrange(number):
            if iteration == 0 and index ==0:
                theta = labels[index]*feature_matrix[index]
                theta_0 += labels[index]
            else:
                (theta, theta_0) = perceptron_single_step_update(feature_matrix[index], labels[index], theta, theta_0)
    return (theta, theta_0)

def passive_aggressive_single_step_update(feature_vector, label, L, current_theta, current_theta_0):
    """
    Section 1.5
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the passive-aggressive algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the passive-aggressive
            algorithm parameters.
        current_theta - The current theta being used by the passive-aggressive
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            passive-aggressive algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    x1 = float(hinge_loss([feature_vector], [label], current_theta, current_theta_0)/(np.linalg.norm(feature_vector))**2)
    x2 = float(1./L)

    n = min(x1, x2)
    theta = current_theta + n*label*feature_vector
    theta_0 = current_theta_0 + n*label

    return (theta, theta_0)

def average_perceptron(feature_matrix, labels, T):
    """
    Section 1.6
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    
    number = len(feature_matrix)
    total = feature_matrix.size
    theta = np.zeros(total/number)
    theta_0 = 0.0
    theta_sum = theta
    theta_0_sum = theta_0

    for iteration in xrange(T):
        for index in xrange(number):
            if iteration == 0 and index ==0:
                theta = labels[index]*feature_matrix[index]
                theta_0 += labels[index]
            else:
                (theta, theta_0) = perceptron_single_step_update(feature_matrix[index], labels[index], theta, theta_0)

            theta_sum += theta
            theta_0_sum += theta_0
    i = float(T * number)
    theta_sum *= 1/i
    theta_0_sum *= 1/i

    return (theta_sum, theta_0_sum)

def average_passive_aggressive(feature_matrix, labels, T, L):
    """
    Section 1.6
    Runs the average passive-agressive algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the passive-agressive
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    
    number = len(feature_matrix)
    total = feature_matrix.size
    theta = np.zeros(total/number)
    theta_0 = 0.0
    theta_sum = theta
    theta_0_sum = theta_0

    for iteration in xrange(T):
        for index in xrange(number):
            (theta, theta_0) = passive_aggressive_single_step_update(feature_matrix[index], labels[index], L, theta, theta_0)
            theta_sum += theta
            theta_0_sum += theta_0

    i = float(T*number)
    theta_sum *= 1/i
    theta_0_sum *= 1/i

    return (theta_sum, theta_0_sum)

### Part II


def classify(feature_matrix, theta, theta_0):
    """
    Section 2.8
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is the predicted
    classification of the kth row of the feature matrix using the given theta
    and theta_0.
    """
    final = np.ndarray(0)
    for index in xrange(len(feature_matrix)):
        vector = feature_matrix[index]
        classify = np.dot(vector,theta) + theta_0
        if classify > 0:
            final = np.append(final, [1])
        else: 
            final = np.append(final, [-1])
    return final

def perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Section 2.9
    Trains a linear classifier using the perceptron algorithm with a given T
    value. The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the perceptron algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    (theta, theta_0) = perceptron(train_feature_matrix, train_labels, T)
    trained = classify(train_feature_matrix, theta, theta_0)
    data = classify(val_feature_matrix, theta, theta_0)

    train = 0
    data_ac = 0
    length1 = len(trained)
    length2 = len(data)
    for index in xrange(length1):
        if trained[index] == train_labels[index]:
            train +=1
    for index in xrange(length2):
        if data[index] == val_labels[index]:
            data_ac += 1
    return (float(train)/length1, float(data_ac)/length2)

def average_perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Section 2.9
    Trains a linear classifier using the average perceptron algorithm with
    a given T value. The classifier is trained on the train data. The
    classifier's accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the average perceptron
            algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """

    (theta, theta_0) = average_perceptron(train_feature_matrix, train_labels, T)
    trained = classify(train_feature_matrix, theta, theta_0)
    data = classify(val_feature_matrix, theta, theta_0)

    train = 0
    data_ac = 0
    length1 = len(trained)
    length2 = len(data)
    for index in xrange(length1):
        if trained[index] == train_labels[index]:
            train +=1
    for index in xrange(length2):
        if data[index] == val_labels[index]:
            data_ac += 1
    return (float(train)/length1, float(data_ac)/length2)

def average_passive_aggressive_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T, L):
    """
    Section 2.9
    Trains a linear classifier using the average passive aggressive algorithm
    with given T and L values. The classifier is trained on the train data.
    The classifier's accuracy on the train and validation data is then
    returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the average passive
            aggressive algorithm.
        L - The value of L to use for training with the average passive
            aggressive algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """

    (theta, theta_0) = average_passive_aggressive(train_feature_matrix, train_labels, T, L)
    trained = classify(train_feature_matrix, theta, theta_0)
    data = classify(val_feature_matrix, theta, theta_0)

    train = 0
    data_ac = 0
    length1 = len(trained)
    length2 = len(data)
    for index in xrange(length1):
        if trained[index] == train_labels[index]:
            train +=1
    for index in xrange(length2):
        if data[index] == val_labels[index]:
            data_ac += 1
    return (float(train)/length1, float(data_ac)/length2)

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    stop = open("stopwords.txt","r")
    swords = []
    for word in stop:
        swords.append(word.rstrip())

    dictionary = {} # maps word to unique index
    word_count = {}
    for text in texts:
        word_list = extract_words(text)
        copy = word_list
        for word in copy:
            if word in swords:
                word_list.remove(word)
        for word in word_list:
            if word not in word_count:
                word_count[word] = 1
            else: 
                word_count[word] += 1
            if word not in dictionary and word_count[word] > 6:
                dictionary[word] = len(dictionary)
    return dictionary

def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix

def extract_additional_features(reviews):
    """
    Section 3.12
    Inputs a list of string reviews
    Returns a feature matrix of (n,m), where n is the number of reviews
    and m is the total number of additional features of your choice

    YOU MAY CHANGE THE PARAMETERS
    """
    return np.ndarray((len(reviews), 0))

def extract_final_features(reviews, dictionary):
    """
    Section 3.12
    Constructs a final feature matrix using the improved bag-of-words and/or additional features
    """
    bow_feature_matrix = extract_bow_feature_vectors(reviews,dictionary)
    additional_feature_matrix = extract_additional_features(reviews)
    return np.hstack((bow_feature_matrix, additional_feature_matrix))

def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """

    return (preds == targets).mean()
