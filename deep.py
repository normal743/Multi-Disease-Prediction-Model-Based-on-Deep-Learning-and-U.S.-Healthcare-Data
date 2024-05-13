import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error

# 读取数据
data = pd.read_csv('/content/drive/MyDrive/seem/deep/combine.csv')
data = data.fillna(data.median())  # 使用 median 补充缺失值

# 数据预处理
X = data.iloc[:, 1:-4].values  # 特征变量
y = data.iloc[:, -4:].values  # 目标变量
y = np.abs(y)
# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 分别训练四个模型
for i in range(4):
    print(f"Training model for Disease {i+1}...")

    # 构建模型
    with tf.device('/gpu:0'):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # 添加 early stopping
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train[:, i], epochs=100, batch_size=32, validation_data=(X_test, y_test[:, i]), callbacks=[es])

    # 评估模型
    y_pred = np.round(model.predict(X_test))
    mae = mean_absolute_error(np.round(y_test[:, i]), np.round(y_pred[:, 0]))
    print(f"MAE for Disease {i+1}: {mae:.2f}")

    # 保存模型
    model.save(f'Edisease{i+1}_model5.h5')
