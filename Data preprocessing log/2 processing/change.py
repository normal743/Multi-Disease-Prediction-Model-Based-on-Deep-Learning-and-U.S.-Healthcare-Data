import pandas as pd

# 读取 CSV 文件
print("reading the file")
df1 = pd.read_csv("/Users/mannormal/Desktop/seem/archive-5/Nhanes/Nhanes_2005_2006.csv")
df2 = pd.read_csv("/Users/mannormal/Desktop/seem/combined_data.csv")
print("processing the file")

# 获取 df1 的所有列名称
columns_df1 = df1.columns

# 初始化替换计数器和新增计数器
replace_count = 0
add_count = 0
not_matched_columns = []

# 逐个比较并替换
for column in columns_df1:
    found_match = False
    for index, value in enumerate(df2["Variable Name"]):
        if value == column:
            new_value = df2.at[index, "Variable Description"]
            df1[column][0] = new_value
            replace_count += 1
            found_match = True
            break
    if not found_match:
        add_count += 1
        not_matched_columns.append(column)
        print(f"未找到 '{column}' 的匹配值，在第一行新增 '{column}' 的值为 '{df2.iloc[0]['Variable Description']}'。")



# 输出新增总数
print(f"共新增 {add_count} 个值。")
if not_matched_columns:
    not_matched_df = pd.DataFrame({"Column Name": not_matched_columns})
    not_matched_df.to_csv("notmatch.csv", index=False)
    print("未找到匹配的列名已保存到 notmatch.csv 文件中。")
# 保存结果到 c.csv
df1.to_csv("c.csv", index=False)
print("结果已保存到 c.csv 文件中。")
