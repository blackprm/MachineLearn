from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection  import GridSearchCV
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split

digits = ds.load_digits()
X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size = 0.2, random_state=666)

param_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors': [i for i in range(1, 11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 7)]
    }
]
knn_clf = KNeighborsClassifier()
''' 
n_jobs : 计算机使用多少内核进行搜索
verbose : 是否进行输出日志
'''
grid_search = GridSearchCV(knn_clf, param_grid, n_jobs= -1, verbose = 100)

grid_search.fit(X_train, Y_train)