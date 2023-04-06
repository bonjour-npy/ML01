import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
from sklearn import tree
import graphviz

# 读入数据
fr = open('glass-lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate', 'type']
lens = pd.DataFrame.from_records(lenses, columns=lensesLabels)
# print(lens)

# 哑变量处理
dummy = pd.get_dummies(lens[['age', 'prescript', 'astigmatic', 'tearRate']])
# 水平合并数据集和哑变量的数据集
lens = pd.concat([lens, dummy], axis=1)
# 删除原始的 age, prescript, astigmatic 和 tearRate 变量
lens.drop(['age', 'prescript', 'astigmatic', 'tearRate'], inplace=True, axis=1)
lens.head()

X_train, X_test, y_train, y_test = model_selection.train_test_split(lens.loc[:, 'age_pre':'tearRate_reduced'],
                                                                    lens.type, test_size=0.25, random_state=1234)

# 构建分类决策树
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train, y_train)

# 对test集进行预测
predictedY = clf.predict(X_test)

# 预测结果
acc = metrics.accuracy_score(y_test, predictedY)
print("accuracy is: {0}".format(acc))

# 绘制
dot_data = tree.export_graphviz(clf, out_file=None, class_names=lensesLabels[0:-1], filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.save("./id3_decision_tree.dot")
