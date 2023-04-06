import sklearn.preprocessing
import torch
import pandas as pd
import torch.nn as nn
import sklearn


def data_loader(data_array, batch_size, is_train=True):
    data_set = torch.utils.data.TensorDataset(*data_array)
    return torch.utils.data.DataLoader(data_set, batch_size, shuffle=is_train)


def normalization(data_array):
    return (data_array - torch.min(data_array)) / (torch.max(data_array) - torch.min(data_array))


data = pd.read_excel("Predict to Profit.xlsx")
# print("original data is:\n{0}".format(data))

# 对离散型特征做独热编码处理
label = sklearn.preprocessing.LabelEncoder()
state_label = label.fit_transform(data["State"])
# print(state_label)
state_label = state_label.reshape(len(state_label), 1)
state_encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False)
state_encoded = state_encoder.fit_transform(state_label)
# print("one-hot encoding of state is:\n{0}".format(state_encoded))

# 特征数据处理
x = torch.concat((torch.tensor([list(data["RD_Spend"])]).reshape(-1, 1),
                  torch.tensor([list(data["Administration"])]).reshape(-1, 1)), -1)
x = torch.concat((x, torch.tensor([list(data["Marketing_Spend"])]).reshape(-1, 1)), -1)
x = torch.concat((x, torch.tensor(state_encoded)), -1)
y = torch.tensor(list(data["Profit"])).reshape(-1, 1)

# 以7:3的比例划分训练集和验证集
index = len(x) // 10 * 7
train_x = x[0:index].to(torch.float32)
train_y = y[0:index].to(torch.float32)
eval_x = x[index:].to(torch.float32)
eval_y = y[index:].to(torch.float32)
original_train_x = train_x.clone().detach()
original_train_y = train_y.clone().detach()

# 对数据进行线性归一化处理
for i in range(train_x.shape[1]):
    train_x[:, i] = normalization(train_x[:, i])
train_y[:, 0] = normalization(train_y[:, 0])
for i in range(eval_x.shape[1]):
    eval_x[:, i] = normalization(eval_x[:, i])
eval_y[:, 0] = normalization(eval_y[:, 0])
print("\nThe shape of train dataset:\nx: {0}, y: {1}".format(train_x.shape, train_y.shape))
print("\nThe shape of eval dataset:\nx: {0}, y: {1}".format(eval_x.shape, eval_y.shape))

# 使用dataset和dataloader定义数据枚举器
batch_size = 5
train_data_iter = data_loader((train_x, train_y), batch_size=batch_size)

# 定义线性模型
net = nn.Sequential(nn.Linear(in_features=train_x.shape[1], out_features=train_y.shape[1]))
# 随机初始化线性模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.normal_(0, 0.01)
loss = nn.MSELoss()
trainer = torch.optim.Adam(net.parameters(), lr=0.03)
epochs = 5000

# 开始训练
for epoch in range(epochs):
    for x, y in train_data_iter:
        trainer_loss = loss(net(x), y)
        trainer_loss.backward()
        trainer.step()
        trainer.zero_grad()
    print("\nfor epoch {0}, train loss is {1}, eval loss is {2}".format(epoch + 1, loss(net(train_x), train_y),
                                                                        loss(net(eval_x), eval_y)))
