import logging
import nltk
import os
import pickle
import pymysql
import sqlite3
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn_evaluation import ClassifierEvaluator

from classification import conf


nltk.download('stopwords')
LOG = logging.getLogger(__name__)
ENG_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
NON_ENG_STOPWORDS = set(nltk.corpus.stopwords.words()) - ENG_STOPWORDS


def is_english(text):
    """Detect language is English or not (Vietnamese)"""
    text = text.lower()
    words = set(nltk.wordpunct_tokenize(text))
    return len(words & ENG_STOPWORDS) > len(words & NON_ENG_STOPWORDS)


def get_stop_words(is_eng=True):
    """Get language's stopwords

    :param boolean is_eng: True if English.
    """
    if is_eng:
        return 'english'
    else:
        with open(conf.VN_SW) as f:
            vn_stopwords = f.read().splitlines()
        return vn_stopwords


def bag_of_words(is_eng=True):
    """Bag of words vectorize.

    :param boolean is_eng: True if English.
    """
    return CountVectorizer(stop_words=get_stop_words(is_eng),
                           token_pattern=r'\b[^\W\d_]+\b')


def td_idf(is_eng=True):
    """Tf Idf vectorize.

    :param boolean is_eng: True if English.
    """
    return TfidfVectorizer(stop_words=get_stop_words(is_eng),
                           token_pattern=r'\b[^\W\d_]+\b')


def get_data_from_db(query, rows, engine=conf.DEFAULT_DB_ENGINE):
    """Connect to database and get data from it.

    :param str query: sql query.
    :param int rows: number of rows.
    :param str engine: database engine.
                       default is MySQL.
    :rtype: list
    """
    # Initialize `connection` var to None. In case we could not
    # create a connection to the database(for ex the disk is full)
    # we would not have a connection var defined.
    connection = None
    try:
        if engine.lower() == 'mysql':
            connection = pymysql.connect(host=conf.MYSQL_HOST,
                                         port=conf.MYSQL_PORT,
                                         user=conf.MYSQL_USER,
                                         password=conf.MYSQL_PASSWORD,
                                         db=conf.MYSQL_DB,
                                         charset='utf8mb4',
                                         cursorclass=pymysql.cursors.
                                         DictCursor)
        elif engine.lower() == 'sqlite':
            connection = sqlite3.connect(conf.SQLITE_DB)
        LOG.info('Connected to {}!' . format(engine))

        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchmany(rows)
        LOG.info('Get random data from database.')

        contents = []
        labels = []
        if engine.lower() == 'mysql':
            for r in result:
                contents.append(r['content'])
                labels.append(r['label'])
        else:
            for r in result:
                contents.append(r[0].strip())
                labels.append(r[1])
        return (contents, labels)
    except Exception as e:
        LOG.exception('Failed when connecting to database: {}. ' . format(e))
    finally:
        if connection:
            LOG.info('Close database connection.')
            connection.close()


def save(obj, path):
    """Save Classifier object to pickle file."""
    if os.path.isfile(path):
        LOG.info('File existed! Use load() method.')
    else:
        pickle.dump(obj, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)


def load(path):
    """Load Classifier object from pickle file"""
    if not os.path.isfile(path):
        LOG.info('File doesnt existed!')
        raise IOError()
    else:
        return pickle.load(open(path, 'rb'))


def split_train_test(X, y, estimator, test_size=0.4):
    LOG.info('Evaluate with test_size is %2.0f%%' % (test_size * 100))
    # Split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    y_score = estimator.predict_proba(X_test)
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_score': y_score
    }
    return result


def evaluate(X, y, estimator, test_size=0.4, confusion=False):
    """Evaluate algorithm.
    :param numpy matrix X: dataset.
    :param numpy matrix y: label matrix.
    :param float test_size: testset's size/alldataset's size.
    :param boolean confusion: use confusion matrix or not.
    """
    result = split_train_test(X, y, estimator, test_size)

    if not confusion:
        return classification_report(result['y_test'], result['y_pred'])
    else:
        return confusion_matrix(result['y_test'], result['y_pred'])


def visualize_with_plot(X, y, estimator, test_size=0.4):
    result = split_train_test(X, y, estimator, test_size)
    target_names = ['business', 'entertainment', 'health', 'politics',
                    'sports', 'technology']
    plot.confusion_matrix(result['y_test'], result['y_pred'],
                          target_names=target_names)


def visualize(X, y, estimator, test_size=0.4, html=False):
    result = split_train_test(X, y, estimator, test_size)
    target_names = ['business', 'entertainment', 'health', 'politics',
                    'sports', 'technology']
    ce = ClassifierEvaluator(estimator, result['y_test'], result['y_pred'],
                            result['y_score'], target_names=target_names)
    if html:
        template = '''
            # Report
            {estimator_type}
            {date}
            {confusion_matrix}
            {roc}
            {precision_recall}
            '''

        ce.generate_report(template, path='report.html')
    else:
        ce.confusion_matrix
