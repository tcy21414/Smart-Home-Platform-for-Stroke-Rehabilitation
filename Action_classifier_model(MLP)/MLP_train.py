import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras import Input


# 设置全局字体为 Arial，字体大小为 8 号
plt.rcParams.update({'font.family': 'Arial', 'font.size': 25})

# 加载预处理后的数据
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# 将 y_train 转换为独热编码
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# 定义全连接模型
model = Sequential()

# 使用 Input 层定义输入形状
model.add(Input(shape=(X_train_flattened.shape[1],)))

# 第一层全连接层，输入形状为展平后的特征数量
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

# 第二层全连接层
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

# 第三层全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

# 输出层，使用 softmax 激活函数，输出类别数
model.add(Dense(len(np.unique(y_train)), activation='softmax'))

# 编译模型
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 编译模型，使用 Adam 优化器和交叉熵损失函数
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型，去掉了验证数据
#history = model.fit(X_train_flattened, y_train_encoded, epochs=50, batch_size=32)
# 假设有测试集 X_test 和 y_test
history = model.fit(X_train_flattened, y_train_encoded, epochs=200, batch_size=64, validation_data=(X_test_flattened, y_test_encoded))

# 获取训练历史（损失和准确度）
history_dict = history.history

# 保存训练好的模型
model.save('pose_classification_dense_model.h5')

