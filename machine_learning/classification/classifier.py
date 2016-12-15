import logging
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from classification import conf
from classification import utils


LOG = logging.getLogger(__name__)


class Classifier(object):

    """Classifier class"""

    def __init__(self, algorithm, text_extract_type, is_eng=True):
        """
        :param str algorithm: choosen algorithm will be used to classify.
        :param str text_extract_type: text extraction (Bag of Words & TF_IDF).
        :param boolean is_eng: language can be eng or vi.
        """
        self.algorithm = algorithm
        self.text_extract_type = text_extract_type
        self.is_eng = is_eng
        self.estimator = None
        self.pickle_file = conf.SAVED_DIR + self.algorithm + \
            '_' + self.text_extract_type + '_' + \
            ''.join('eng' if self.is_eng else 'vi') + '.pickle'
        self.train()

    def load_train_set(self):
        """Load train set from database."""
        query = conf.QUERY
        return utils.get_data_from_db(query, conf.NUMBER_ROWS)

    def word_to_vector(self):
        """Convert from text to vector."""
        contents, labels = self.load_train_set()

        if self.text_extract_type == 'Bag Of Words':
            LOG.info('Convert contents to BOW.')
            self.vectorizer = utils.bag_of_words()
        else:
            LOG.info('Convert contents to TF-IDF.')
            self.vectorizer = utils.td_idf()

        self.X = self.vectorizer.fit_transform(contents)
        self.y = np.array(labels)

    def _tuning_parameters(self):
        """Tuning the hyper-parameters of an estimator."""

        if self.algorithm == 'SVM':
            tmp_estimator = SVC(C=1, probability=True)
            pre_tune_params = conf.SVM_PARAMS
        else:
            tmp_estimator = KNeighborsClassifier()
            pre_tune_params = conf.KNN_PARAMS

        # Tuning estimator, modify n_jobs and pre_dispatch depend on
        # your machine, which run this. More details, plz read sklearn
        # documentation about GridSearchCV.
        grid_search = GridSearchCV(tmp_estimator, pre_tune_params,
                                   cv=5, n_jobs=-1, pre_dispatch='0.5*n_jobs')
        grid_search.fit(self.X, self.y)
        LOG.info('Tuning the hyper-parameters of an {} estimator' .
                 format(self.algorithm))
        self.estimator = grid_search.best_estimator_

    def train(self):
        LOG.info('Training...')
        self.word_to_vector()
        if not self.estimator:
            self._tuning_parameters()

    def predict(self, content):
        content = self.vectorizer.transform(content)
        return self.estimator.predict(content)
