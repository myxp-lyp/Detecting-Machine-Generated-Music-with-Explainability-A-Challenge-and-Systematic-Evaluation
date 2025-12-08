import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
import time
from  tqdm import tqdm
data_dir = "./porcess_dataset"  # 修改为你的.npy文件路径
X = []  # 特征
y = []  # 标签

# 遍历文件夹并加载数据
for filename in tqdm(os.listdir(data_dir)):
    if filename.endswith('.npy'):
        file_path = os.path.join(data_dir, filename)
        data = np.load(file_path)  # 加载.npy文件
        X.append(data)
        
        # 获取标签，假设文件名开头是0或1
        label = int(filename[0])
        y.append(label)

X = np.array(X)
y = np.array(y)
X = X.reshape(X.shape[0], -1)  # 将每张图片展平为一维向量
print(f"Reshaped X shape: {X.shape}")

print(f"Loaded {len(X)} samples with labels.")

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
start_time = time.perf_counter()
# 使用传统的 SVM 分类器
svm = SVC(kernel='poly', degree=2, coef0=1)  # 这里使用了二次核函数
svm.fit(X_train, y_train)
end_time = time.perf_counter()

# 进行预测
predictions = svm.predict(X_test)

training_time = end_time - start_time
print(f"Training Time: {training_time:.4f} seconds")
# 输出准确度
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# 计算 F1 指标
f1 = f1_score(y_test, predictions)
print(f"F1 Score: {f1:.2f}")