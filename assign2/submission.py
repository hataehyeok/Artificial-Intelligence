#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *


############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, touching, quite, impressive, not, boring
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'so': 1, 'touching': 1, 'quite': 0, 'impressive': 0, 'not': -1, 'boring': -1}
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code , but don't worry if you deviate from this)
    return dict(Counter(x.split()))
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    def update_weights(weights, eta, x, y, featureExtractor):
        phi = featureExtractor(x)
        sigma = sigmoid(dotProduct(weights, phi))
        prob = sigma if y == 1 else (1 - sigma)
        scale = (eta * y * sigma * (1 - sigma)) / prob
        increment(weights, scale, phi)

    for _ in range(numIters):
        [update_weights(weights, eta, x, y, featureExtractor) for x, y in trainExamples]
    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: ngram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER (our solution is 12 lines of code, but don't worry if you deviate from this)
    arr = []
    words = x.split()

    for i in range(0, len(words) - n + 1):
        word_sum = str()
        for j in range(i, n + i):
            if j != n + i - 1:
                word_sum += (words[j] + " ")
            else:
                word_sum += words[j]
        arr.append(word_sum)
    
    phi = dict(Counter(arr))
    # END_YOUR_ANSWER
    return phi


############################################################
# Problem 3a: k-means exercise
############################################################

def problem_3a_1():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -2, 'mu_y': 0}, {'mu_x': 3, 'mu_y': 0})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'mu_x': -0.5, 'mu_y': 1.5}, {'mu_x': 3, 'mu_y': 1.5}
    # END_YOUR_ANSWER

def problem_3a_2():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -1, 'mu_y': -1}, {'mu_x': 2, 'mu_y': 3})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'mu_x': -1, 'mu_y': 0}, {'mu_x': 2, 'mu_y': 2}
    # END_YOUR_ANSWER

############################################################
# Problem 3: k-means implementation
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_ANSWER (our solution is 40 lines of code, but don't worry if you deviate from this)
    def dist(idx_c, idx_p):
        return centroids_squared[idx_c] + squared_vector[idx_p] - 2 * dotProduct(centroids[idx_c], examples[idx_p]) 
    
    squared_vector = [dotProduct(example, example) for example in examples]
    centroids = random.sample(examples, K)
    centroids_squared = [dotProduct(centroid, centroid) for centroid in centroids]
    
    assignments = {}

    for _ in range(maxIters):
        temp = [min(range(K), key = lambda j: dist(j, i)) for i in range(len(examples))]
        if temp == assignments:
            break
        assignments = temp

        means = [[{}, 0] for _ in range(K)]
        for i, assignment in enumerate(assignments):
            increment(means[assignment][0], 1, examples[i])
            means[assignment][1] += 1
        
        for i, (mean, size) in enumerate(means):
            if size > 0:
                for k, v in mean.items():
                    mean[k] = v / size
            centroids[i] = mean
            centroids_squared[i] = dotProduct(mean, mean)
    
    cast = sum(dist(assignments[i], i) for i in range(len(examples)))
    
    return centroids, assignments, cast
    # END_YOUR_ANSWER

