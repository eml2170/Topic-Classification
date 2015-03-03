# Topic-Classification
Uses vocabulary to classify the topic of text

This work was done for Daniel Hsu's Machine Learning class at Columbia Spring 2015.

It uses the "20 Newsgroups data set" http://scikit-learn.org/stable/datasets/twenty_newsgroups.html. The feature vectors and labels are stored in the matlab files as "data" and "labels," and the tests feature vectors are and labels are stored as "testdata" and "testlabels."

The representation of a message is a (sparse) binary vector in {0,1}^d (for d = 61188) that indicates the
words that are present in the message. If the j-th entry in the vector is 1, it means the message
contains the word that is given on the j-th line of the text file news.vocab. The class labels are
Y = {1, 2, . . . , 20}, where the mapping from classes to newsgroups is in the file data/news.groups.

I implemented a Bernoulli Naive Bayes classifier as well as the Perceptron (both online and averaged). On test, average perceptron runs best, especially when the hypothesis passes through the data seen so far multiple times.