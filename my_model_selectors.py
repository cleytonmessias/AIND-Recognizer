import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant
    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("-inf")
        best_model = None
        num_features = self.X.shape[1]

        for i_components in range(self.min_n_components, self.max_n_components+1):
            try:
                new_model = self.base_model(i_components)

                logL = new_model.score(self.X, self.lengths)
                logN = np.log(len(self.X))

                initial_state = i_components

                transition_probs = i_components * (i_components-1)
                emission_probs = i_components * num_features * 2

                p = initial_state + transition_probs + emission_probs

                new_score = -2 * logL + p * logN

                best_score, best_model = max((best_score, best_model), (new_score, new_model))

            except:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("-inf")
        best_model = None

        num_features = self.X.shape[1]

        for i_components in range(self.min_n_components, self.max_n_components +1):
            try:
                new_model = self.base_model(i_components)

                logL = new_model.score(self.X, self.lengths)

                partial_score = 0
                count = 0

                for word in self.words:
                    if word != self.this_word:
                        new_X, new_leghts = self.hwords[word]

                        try:
                            partial_score += hmm_model.score(new_X, new_leghts)
                            count += 1
                        except:
                            pass
                if count > 0:
                    logAll = partial_score/count
                else:
                    logAll = 0

                new_score = logL - logAll
                best_score, best_model = max((best_score,best_model),(new_score,new_model))
            except:
                pass

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def splits(self):
        return min(3, len(self.sequences))

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("-inf")
        best_model = None


        if len(self.sequences) < 2:
            return best_model

        kf = KFold(n_splits=self.splits())

        for i_components in range(self.min_n_components, self.max_n_components +1 ):
            partial_score = 0
            count = 0

            for cv_train_idx, cv_test_idx in kf.split(self.sequences):
                new_model = None
                try:
                    train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                    test_X, test_lenghts = combine_sequences(cv_test_idx, self.sequences)

                    new_model = GaussianHMM(n_components = i_components, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(train_X, train_lengths)

                    partial_score += hmm_model.score(test_X, test_lenghts)
                    count += 1
                except:
                    pass

            if count > 0:
                new_score = partial_score / count
            else:
                new_score = 0

            if new_score > best_score:
                best_score = new_score
                best_model = new_model

        return best_model
