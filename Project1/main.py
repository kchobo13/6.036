import project1 as p1
import utils

#-------------------------------------------------------------------------------
# Data loading. There is no need to edit code in this section.
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

dictionary = p1.bag_of_words(train_texts)

train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)

#-------------------------------------------------------------------------------
#Section 1.7
#-------------------------------------------------------------------------------
# toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')

# T = 5
# L = 10

# thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
# thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
# thetas_avg_pa = p1.average_passive_aggressive(toy_features, toy_labels, T, L)

# def plot_toy_results(algo_name, thetas):
#    utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)

# plot_toy_results('Perceptron', thetas_perceptron)
# plot_toy_results('Average Perceptron', thetas_avg_perceptron)
# plot_toy_results('Average Passive-Aggressive', thetas_avg_pa)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Section 2.9.b
#-------------------------------------------------------------------------------
# T = 5
# L = 1

# pct_train_accuracy, pct_val_accuracy = \
# p1.perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T)
# print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
# print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

# avg_pct_train_accuracy, avg_pct_val_accuracy = \
# p1.average_perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T)
# print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
# print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

# avg_pa_train_accuracy, avg_pa_val_accuracy = \
# p1.average_passive_aggressive_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
# print("{:50} {:.4f}".format("Training accuracy for average passive-aggressive:", avg_pa_train_accuracy))
# print("{:50} {:.4f}".format("Validation accuracy for average passive-aggressive:", avg_pa_val_accuracy))
##-------------------------------------------------------------------------------
#
#
#-------------------------------------------------------------------------------
# Section 2.10
#-------------------------------------------------------------------------------
# data = (train_bow_features, train_labels, val_bow_features, val_labels)

# # values of T and lambda to try
# Ts = [1,5,10,15,20]
# Ls = [1,5,10,20,50,100]

# pct_tune_results = utils.tune_perceptron(Ts, *data)
# avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)

# # fix values for L and T while tuning passive-aggressive T and L, respective
# best_L = 20
# best_T = 5

# avg_pa_tune_results_T = utils.tune_passive_aggressive_T(best_L, Ts, *data)
# avg_pa_tune_results_L = utils.tune_passive_aggressive_L(best_T, Ls, *data)

# utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
# utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
# utils.plot_tune_results('Avg Passive-Aggressive', 'T', Ts, *avg_pa_tune_results_T)
# utils.plot_tune_results('Avg Passive-Aggressive', 'L', Ls, *avg_pa_tune_results_L)
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 2.11a
#
# Call one of the accuracy functions that you wrote in part 2.9.a and report
# the hyperparameter and accuracy of your best classifier on the test data.
# The test data has been provided as test_bow_features and test_labels.
#-------------------------------------------------------------------------------
# print p1.average_passive_aggressive_accuracy(train_bow_features, test_bow_features, train_labels, test_labels, 5, 20)
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 2.11b
#
# Assign to best_theta, the weights (and not the bias!) learned by your most
# accurate algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------
# best_theta = p1.average_passive_aggressive(train_bow_features, train_labels, 5, 20)[0]
# wordlist   = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
# sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
# print("Most Explanatory Word Features")
# print(sorted_word_features[:10])
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 3.12
#
# After constructing a final feature representation, use code similar to that in
# sections 2.9b and 2.10 to assess its performance on the validation set.
# You may use your best classifier from before as a baseline.
# When you are satisfied with your features, evaluate their accuracy on the test
# set using the same procedure as in section 2.11a.
#-------------------------------------------------------------------------------
dictionary = p1.bag_of_words(train_texts)

train_final_features = p1.extract_final_features(train_texts, dictionary)
val_final_features   = p1.extract_final_features(val_texts, dictionary)
test_final_features  = p1.extract_final_features(test_texts, dictionary)

T = 5
L = 20

pct_train_accuracy, pct_val_accuracy = \
p1.perceptron_accuracy(train_final_features,val_final_features,train_labels,val_labels,T=T)
print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

avg_pct_train_accuracy, avg_pct_val_accuracy = \
p1.average_perceptron_accuracy(train_final_features,val_final_features,train_labels,val_labels,T=T)
print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

avg_pa_train_accuracy, avg_pa_val_accuracy = \
p1.average_passive_aggressive_accuracy(train_final_features,val_final_features,train_labels,val_labels,T=T,L=L)
print("{:50} {:.4f}".format("Training accuracy for average passive-aggressive:", avg_pa_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for average passive-aggressive:", avg_pa_val_accuracy))

# data = (train_bow_features, train_labels, val_bow_features, val_labels)

# # values of T and lambda to try
# Ts = [1,5,10,15,20]
# Ls = [1,5,10,20,50,100]

# pct_tune_results = utils.tune_perceptron(Ts, *data)
# avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)

# # fix values for L and T while tuning passive-aggressive T and L, respective
# best_L = 20
# best_T = 5

# avg_pa_tune_results_T = utils.tune_passive_aggressive_T(best_L, Ts, *data)
# avg_pa_tune_results_L = utils.tune_passive_aggressive_L(best_T, Ls, *data)

# utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
# utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
# utils.plot_tune_results('Avg Passive-Aggressive', 'T', Ts, *avg_pa_tune_results_T)
# utils.plot_tune_results('Avg Passive-Aggressive', 'L', Ls, *avg_pa_tune_results_L)
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 3.13
#
# Modify the code below to extract your best features from the submission data
# and then classify it using your most accurate classifier.
#-------------------------------------------------------------------------------
# submit_texts = [sample['text'] for sample in utils.load_data('reviews_submit.tsv')]
#
# # 1. Extract your preferred features from the train and submit data
# dictionary = p1.bag_of_words(submit_texts)
# train_final_features = p1.extract_final_features(train_texts, dictionary)
# submit_final_features = p1.extract_final_features(submit_texts, dictionary)
#
# # 2. Train your most accurate classifier
# final_thetas = p1.perceptron(train_final_features, train_labels, T=1)
#
# # 3. Classify and write out the submit predictions.
# submit_predictions = p1.classify(submit_final_features, *final_thetas)
# utils.write_predictions('reviews_submit.tsv', submit_predictions)
#-------------------------------------------------------------------------------
