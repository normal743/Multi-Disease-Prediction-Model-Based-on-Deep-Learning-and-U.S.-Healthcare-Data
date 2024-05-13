import pandas as pd
from autogluon.tabular import TabularPredictor

# 读取数据
data = pd.read_csv('/content/drive/MyDrive/seem/8K line/combine.csv')

# 填充缺失值
data = data.fillna(data.median())

# 数据预处理
X = data.iloc[:, 1:-4]  # 特征变量
y_columns = ['angina', 'WHQ030.1', 'DIQ010.1', 'BPQ020.1']  # 根据输出修改列名
y = data[y_columns]  # 目标变量

# 对每个目标特征训练一个模型
for col in y_columns:
    y_col = y[col]
    predictor = TabularPredictor(label=y_col).fit(X)
    predictor.save_model(f'/path/to/save/model_{col}')
