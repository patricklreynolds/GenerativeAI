
SAVE_PARAMS_EVERY = 5000

import glob
import os.path as op
import pickle
import random

import numpy as np

from utils.gradcheck import gradcheck_naive
from utils.utils import softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """
    ### START CODE HERE
    s = 1 / (1 + np.exp(-x))
    ### END CODE HERE

    return s


def naive_softmax_loss_and_gradient(center_word_vec, outside_word_idx, outside_vectors, dataset):
    """
    Naive Softmax loss & gradient function for word2vec models.
    """
    ### START CODE HERE
    y_hat = softmax(np.dot(outside_vectors, center_word_vec))
    loss = -np.log(y_hat[outside_word_idx])

    y = np.zeros_like(y_hat)
    y[outside_word_idx] = 1

    grad_center_vec = np.dot((y_hat - y), outside_vectors)
    grad_outside_vecs = np.outer((y_hat - y), center_word_vec)
    ### END CODE HERE

    return loss, grad_center_vec, grad_outside_vecs


def get_negative_samples(outside_word_idx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """
    neg_sample_word_indices = [None] * K
    for k in range(K):
        newidx = dataset.sample_token_idx()
        while newidx == outside_word_idx:
            newidx = dataset.sample_token_idx()
        neg_sample_word_indices[k] = newidx
    return neg_sample_word_indices

def neg_sampling_loss_and_gradient(center_word_vec, outside_word_idx, outside_vectors, dataset, K=10):
    """ Negative sampling loss function for word2vec models

    Arguments/Return Specifications: same as naive_softmax_loss_and_gradient
    K is the number of negative samples to take.
    """

    neg_sample_word_indices = get_negative_samples(outside_word_idx, dataset, K)
    indices = [outside_word_idx] + neg_sample_word_indices

    grad_center_vec = np.zeros(center_word_vec.shape)
    grad_outside_vecs = np.zeros(outside_vectors.shape)

    labels = np.array([1] + [-1 for k in range(K)])
    vecs = outside_vectors[indices, :]

    t = sigmoid(vecs.dot(center_word_vec) * labels)
    loss = -np.sum(np.log(t))

    delta = labels * (t - 1)
    grad_center_vec = delta.reshape((1, K + 1)).dot(vecs).flatten()
    grad_outside_vecs_temp = delta.reshape((K + 1, 1)).dot(center_word_vec.reshape(
        (1, center_word_vec.shape[0])))
    for k in range(K + 1):
        grad_outside_vecs[indices[k]] += grad_outside_vecs_temp[k, :]

    return loss, grad_center_vec, grad_outside_vecs

def skipgram(current_center_word, window_size, outside_words, word2ind, center_word_vectors, outside_vectors, dataset, word2vec_loss_and_gradient=naive_softmax_loss_and_gradient):
    """
    Skip-gram model in word2vec.
    """
    loss = 0.0
    grad_center_vecs = np.zeros(center_word_vectors.shape)
    grad_outside_vectors = np.zeros(outside_vectors.shape)

    ### START CODE HERE
    center_word_idx = word2ind[current_center_word]
    center_word_vec = center_word_vectors[center_word_idx]

    for outside_word in outside_words:
        outside_word_idx = word2ind[outside_word]
        current_loss, grad_center, grad_outside = word2vec_loss_and_gradient(center_word_vec, outside_word_idx, outside_vectors, dataset)
        
        loss += current_loss
        grad_center_vecs[center_word_idx] += grad_center
        grad_outside_vectors += grad_outside
    ### END CODE HERE

    return loss, grad_center_vecs, grad_outside_vectors


def word2vec_sgd_wrapper(word2vec_model, word2ind, word_vectors, dataset, window_size, word2vec_loss_and_gradient=neg_sampling_loss_and_gradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(word_vectors.shape)
    N = word_vectors.shape[0]
    center_word_vectors = word_vectors[:int(N / 2), :]
    outside_vectors = word_vectors[int(N / 2):, :]
    for i in range(batchsize):
        window_size_1 = random.randint(1, window_size)
        center_word, context = dataset.get_random_context(window_size_1)

        c, gin, gout = word2vec_model(center_word, window_size_1, context, word2ind, center_word_vectors, outside_vectors, dataset, word2vec_loss_and_gradient)
        loss += c / batchsize
        grad[:int(N / 2), :] += gin / batchsize
        grad[int(N / 2):, :] += gout / batchsize

    return loss, grad


def load_saved_params():
    """
    A helper function that loads previously saved parameters and resets
    iteration start.
    """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        params_file = "saved_params_%d.npy" % st
        state_file = "saved_state_%d.pickle" % st
        params = np.load(params_file)
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None

def save_params(iter, params):
    params_file = "saved_params_%d.npy" % iter
    np.save(params_file, params)
    with open("saved_state_%d.pickle" % iter, "wb") as f:
        pickle.dump(random.getstate(), f)

def sgd(f, x0, step, iterations, postprocessing=None, use_saved=False, PRINT_EVERY=10):
    """ Stochastic Gradient Descent

    Implement the stochastic gradient descent method in this function.

    Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a loss and the gradient
         with respect to the arguments
    x0 -- the initial point to start SGD from
    step -- the step size for SGD
    iterations -- total iterations to run SGD for
    postprocessing -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit length.
    PRINT_EVERY -- specifies how many iterations to output loss

    Return:
    x -- the parameter value after SGD finishes
    """

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000

    if use_saved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    exploss = None

    for iter in range(start_iter + 1, iterations + 1):
        # You might want to print the progress every few iterations.
        loss = None

        ### START CODE HERE
        loss, grad = f(x)
        x -= step * grad
        ### END CODE HERE

        x = postprocessing(x)
        if iter % PRINT_EVERY == 0:
            if not exploss:
                exploss = loss
            else:
                exploss = .95 * exploss + .05 * loss
            print("iter %d: %f" % (iter, exploss))

        if iter % SAVE_PARAMS_EVERY == 0 and use_saved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x
