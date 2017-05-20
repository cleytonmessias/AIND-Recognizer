import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for word_id in range(0, len(test_set.get_all_Xlengths())):

        X, lengths = test_set.get_item_Xlengths(word_id)

        probs = {}
        # best_score = float('-Inf')
        inf_score = float('-Inf')
        guessed_word = None

        for word, model in models.items():
          try:
            score = model.score(X, lengths)
            probs[word] = score
          except:
            probs[word] = inf_score
            pass

        sort = sorted(probs, key=probs.get, reverse=True)

        guesses.append(sort[0])
        probabilities.append(probs)

    return probabilities, guesses
