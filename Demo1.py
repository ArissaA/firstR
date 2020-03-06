from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer


def dictvec():
    dict = DictVectorizer(sparse=False)
    data = dict.fit_transform([{'city': '北京', 'temperature': 100}, {
                              'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}])

    print(dict.get_feature_names())
    print(data)

    print(dict.inverse_transform(data))

    return None


def countvec():

    cv = CountVectorizer()

    data = cv.fit_transform(
        ["life is short,i like python", "life is too long,i dislike python"])

    print(cv.get_feature_names())
    a = data.toarray()
    b = a.sum(axis=0)
    print(a)
    print(b)
    return None


def tfidfvec():
    vv = TfidfVectorizer()

    data = vv.fit_transform(
        ["life is short,i like python", "life is too long,i dislike python"])
    data_array = data.toarray()
    b = vv.get_feature_names()
    print(data_array)
    print(b)
    return None


if __name__ == '__main__':
    # dictvec()
    # countvec()
    tfidfvec()
