# coding=utf-8
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 使用svm 进行分类，返回svm 模型
# data:(sp_train, sp_b, lab_train, lab_b)
def _svm(data, target_names):
    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 创建SVM分类器
    svm_classifier = SVC(kernel='linear')  # 使用线性核函数

    # 训练模型
    svm_classifier.fit(X_train, y_train)

    # 对测试集进行预测
    y_pred = svm_classifier.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.2f}")

    # 打印分类报告
    print("分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 打印混淆矩阵
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    ## 返回分类器，用于下次分类
    return svm_classifier



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
def svm(file_path):
    # 读取 CSV 文件
    
    df = pd.read_csv(file_path,encoding='gbk')

    # # 查看前几行
    # print("查看前 5 行：")
    # print(df.head())
    from boxsers.misc_tools import data_split
    from boxsers.visual_tools import distribution_plot

    # Features extraction: Exports dataframe spectra as a numpy array (value type = float64).
    sp = df.iloc[:, 1:].to_numpy()
    # print ("????")
    # print (sp[6])
    # print (sp.shape)
    # Labels extraction: Export dataframe classes into a numpy array of string values.
    classnames = df['class'].unique()  
    # print (classnames)
    label = df.loc[:, 'class'].values



    # String to integer labels conversion: 
    labelencoder = LabelEncoder()  # Creating instance of LabelEncoder
    lab_int = labelencoder.fit_transform(label)  # 0, 3, 2, ...

    # print("lab_int")
    # print(lab_int)

    # String to binary labels conversion: 
    labelbinarizer = LabelBinarizer()  # Creating instance of LabelBinarizer
    lab_binary = labelbinarizer.fit_transform(label)  # [1 0 0 0] [0 0 0 1] [0 1 0 0], ...


    # Train/Validation/Test sets splitting 
    (sp_train, sp_b, lab_train, lab_b) = data_split(sp, lab_int, b_size=0.5, rdm_ste=None, print_report=False)
    # print ("lable")
    # print  (len(sp_train))
    # print  (len(lab_train))
    # (sp_val, sp_test, lab_val, lab_test) = data_split(sp, label, b_size=0.1, rdm_ste=None, print_report=False)
    data = (sp_train, sp_b, lab_train, lab_b)
    return _svm(data, target_names=classnames)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import datasets
# 使用cnn（一维）进行分类，返回 cnn 模型
# data:(sp_train, sp_b, lab_train, lab_b)
def _cnn(data,target_names):
    # # 将标签转换为one-hot编码
    # y = tf.keras.utils.to_categorical(y, num_classes=3)

    # # 将特征数据转换为适合1D CNN的格式 (样本数, 特征数, 1)
    # X = np.expand_dims(X, axis=-1)

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)
    
    # 将标签转换为one-hot编码
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(target_names))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(target_names))

    # 定义1D CNN模型
    model = models.Sequential([
    layers.Input(shape=(X_train.shape[1], 1)),  # 输入形状 (特征数, 1)
    layers.Conv1D(filters=32, kernel_size=3, activation='relu'),  # 1D卷积层
    layers.MaxPooling1D(pool_size=2),  # 最大池化层
    layers.Flatten(),  # 展平层
    layers.Dense(64, activation='relu'),  # 全连接层
    layers.Dense(len(target_names), activation='softmax')  # 输出层 (3个类别)
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 打印模型摘要
    model.summary()

    print (X_train.shape)
    print (y_train.shape)
    # 训练模型
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

    # 对测试集进行预测
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # 计算准确率
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f"模型准确率: {accuracy:.2f}")

    # 打印分类报告
    print("分类报告:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=target_names))

    # 打印混淆矩阵
    print("混淆矩阵:")
    print(confusion_matrix(y_test_classes, y_pred_classes))
    return model

def cnn(file_path):
    # 读取 CSV 文件
    
    df = pd.read_csv(file_path,encoding='gbk')

    # # 查看前几行
    # print("查看前 5 行：")
    # print(df.head())
    from boxsers.misc_tools import data_split
    from boxsers.visual_tools import distribution_plot

    # Features extraction: Exports dataframe spectra as a numpy array (value type = float64).
    sp = df.iloc[:, 1:].to_numpy()
    # print ("????")
    # print (sp[6])
    # print (sp.shape)
    # Labels extraction: Export dataframe classes into a numpy array of string values.
    classnames = df['class'].unique()  
    # print (classnames)
    label = df.loc[:, 'class'].values



    # String to integer labels conversion: 
    labelencoder = LabelEncoder()  # Creating instance of LabelEncoder
    lab_int = labelencoder.fit_transform(label)  # 0, 3, 2, ...

    # print("lab_int")
    # print(lab_int)

    # String to binary labels conversion: 
    labelbinarizer = LabelBinarizer()  # Creating instance of LabelBinarizer
    lab_binary = labelbinarizer.fit_transform(label)  # [1 0 0 0] [0 0 0 1] [0 1 0 0], ...


    # Train/Validation/Test sets splitting 
    (sp_train, sp_b, lab_train, lab_b) = data_split(sp, lab_int, b_size=0.5, rdm_ste=None, print_report=False)
    # print ("lable")
    # print  (len(sp_train))
    # print  (len(lab_train))
    # (sp_val, sp_test, lab_val, lab_test) = data_split(sp, label, b_size=0.1, rdm_ste=None, print_report=False)
    data = (sp_train, sp_b, lab_train, lab_b)
    return _cnn(data, target_names=classnames)

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def _decision_tree(data=None, target_names=None):

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]


    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)
    
    # 将标签转换为one-hot编码
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(target_names))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(target_names))


    # 创建决策树模型
    clf = DecisionTreeClassifier(random_state=42)

    # 训练模型
    clf.fit(X_train, y_train)

    # 预测结果
    y_pred = clf.predict(X_test)

    # 评估模型性能
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy)

    feature_names = [str(i) for i in range(len(X_test[0]))]
    # print (feature_names)

    # 可视化决策树
    from sklearn.tree import plot_tree
    # print (len(X_train[0]))
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_tree(clf, feature_names=feature_names, class_names=target_names)
    plt.show()


def decision_tree(file_path):
    # 读取 CSV 文件
    
    df = pd.read_csv(file_path,encoding='gbk')

    # # 查看前几行
    # print("查看前 5 行：")
    # print(df.head())
    from boxsers.misc_tools import data_split
    from boxsers.visual_tools import distribution_plot

    # Features extraction: Exports dataframe spectra as a numpy array (value type = float64).
    sp = df.iloc[:, 1:].to_numpy()
    # print ("????")
    # print (sp[6])
    # print (sp.shape)
    # Labels extraction: Export dataframe classes into a numpy array of string values.
    classnames = df['class'].unique()  
    # print (classnames)
    label = df.loc[:, 'class'].values



    # String to integer labels conversion: 
    labelencoder = LabelEncoder()  # Creating instance of LabelEncoder
    lab_int = labelencoder.fit_transform(label)  # 0, 3, 2, ...

    # print("lab_int")
    # print(lab_int)

    # String to binary labels conversion: 
    labelbinarizer = LabelBinarizer()  # Creating instance of LabelBinarizer
    lab_binary = labelbinarizer.fit_transform(label)  # [1 0 0 0] [0 0 0 1] [0 1 0 0], ...


    # Train/Validation/Test sets splitting 
    (sp_train, sp_b, lab_train, lab_b) = data_split(sp, lab_int, b_size=0.5, rdm_ste=None, print_report=False)
    # print ("lable")
    # print  (len(sp_train))
    # print  (len(lab_train))
    # (sp_val, sp_test, lab_val, lab_test) = data_split(sp, label, b_size=0.1, rdm_ste=None, print_report=False)
    data = (sp_train, sp_b, lab_train, lab_b)
    _decision_tree(data,target_names=classnames)
    


    
if __name__ == "__main__":
    file_path = 'food/data/data.csv'  # 替换为你的文件路径
    print ("使用decision_tree")
    decision_tree(file_path)
