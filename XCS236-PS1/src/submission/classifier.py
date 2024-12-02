import torch
from submission.likelihood import log_likelihood

def classification(model, text):
    """
    Classify whether the string `text` is randomly generated or not.
    :param model: The GPT-2 model
    :param text: A tensor of shape (1, T), where T is the length of the text
    :return: True if `text` is a random string. Otherwise return False
    """

    with torch.no_grad():
        ### START CODE HERE ###
        ll = log_likelihood(model, text)  # Calculate log likelihood of the text
        return ll < -300  # Return True if log likelihood is less than -300, indicating it is likely random
        ### END CODE HERE ###
