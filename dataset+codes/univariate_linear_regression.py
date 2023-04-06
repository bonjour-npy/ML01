# 导入第三方模块
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import linear_model

# 导入数据集
income = pd.read_csv(r'line-ext.csv')
# 绘制散点图
sns.lmplot(x='YearsExperience', y='Salary', data=income, ci=None)
# 显示图形
plt.show()

# 样本量
n = income.shape[0]
inputs = income.YearsExperience
labels = income.Salary
# 计算自变量、因变量、自变量平方、自变量与因变量乘积的和
sum_x = income.YearsExperience.sum()
sum_y = income.Salary.sum()
sum_x2 = income.YearsExperience.pow(2).sum()
xy = income.YearsExperience * income.Salary
sum_xy = xy.sum()
# 根据公式计算回归模型的参数
w = (sum_xy - sum_x * sum_y / n) / (sum_x2 - sum_x ** 2 / n)
b = income.Salary.mean() - w * income.YearsExperience.mean()
# 打印出计算结果
print('回归参数 w 的值：', w)
print('回归参数 b 的值：', b)
print('模型表达式：f(x) = {0} * x + {1}'.format(w, b))
# 打印均方误差
predicted_y = income.YearsExperience * w + b
test_y = income.Salary
mse_loss = metrics.mean_squared_error(test_y, predicted_y)
print("MSE Loss is: {0}".format(mse_loss))

# 请给出当自变量 x = 0.8452 时，因变量 y 的预测值
x = 0.8452
y = x * w + b
print("When x equals 0.8452, y equals {0}".format(y))

# Implementation of Statsmodels
# 利用收入数据集,构建回归模型
fit = sm.formula.ols("Salary ~ YearsExperience", data=income).fit()
# 返回模型的参数值
print("w of Statsmodels is {0}".format(fit.params["YearsExperience"]))
print("b of Statsmodels is {0}".format(fit.params["Intercept"]))

# Implementation of Scikit-learn
model = linear_model.LinearRegression()
inputs = np.array(inputs)
labels = np.array(labels)
inputs = np.reshape(inputs, (-1, 1))
labels = np.reshape(labels, (-1, 1))
# print(inputs, labels)
model.fit(inputs, labels)
mse_loss_hat = metrics.mean_squared_error(model.predict(inputs), labels)
print(mse_loss_hat)
print("w of sklearn is: {0}".format(model.coef_))
print("b of sklearn is: {0}".format(model.intercept_))
