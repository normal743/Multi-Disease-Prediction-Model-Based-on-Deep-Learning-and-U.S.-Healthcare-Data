import os
import pandas as pd

def process_files_in_folder(folder_path, appendix2_id, appendix5_id):
    """
    处理文件夹中的文件，将第一行与附录2和附录5中的第一列进行比较，
    如果匹配，则将该行的第二列替换为对应的附录2和附录5中的第二列。

    Parameters:
        folder_path (str): 要处理的文件夹路径。
        appendix2_id (str): 附录2的 ID。
        appendix5_id (str): 附录5的 ID。
    """
    # 读取附录2和附录5数据
    appendix2_data = pd.read_csv(appendix2_id + '.csv')
    appendix5_data = pd.read_csv(appendix5_id + '.csv')

    # 获取附录2和附录5中的第一列和第二列
    appendix2_col1 = appendix2_data.iloc[:, 0]
    appendix2_col2 = appendix2_data.iloc[:, 1]
    appendix5_col1 = appendix5_data.iloc[:, 0]
    appendix5_col2 = appendix5_data.iloc[:, 1]

    # 遍历文件夹中的文件
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            # 读取文件的第一行数据
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            
            # 将第一行与附录2的第一列进行比较
            if first_line in appendix2_col1.values:
                # 找到第一行在附录2中的索引
                idx = appendix2_col1[appendix2_col1 == first_line].index[0]
                # 获取附录2中对应行的第二列数据
                new_value = appendix2_col2[idx]
            # 将第一行与附录5的第一列进行比较
            elif first_line in appendix5_col1.values:
                # 找到第一行在附录5中的索引
                idx = appendix5_col1[appendix5_col1 == first_line].index[0]
                # 获取附录5中对应行的第二列数据
                new_value = appendix5_col2[idx]
            else:
                new_value = None

            # 替换文件的第二行数据
            if new_value is not None:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                lines[1] = new_value + '\n'
                with open(file_path, 'w') as f:
                    f.writelines(lines)

# 示例使用方法
folder_path = 'your_folder_path_here'  # 替换为你的文件夹路径
appendix2_id = 'appendix2_data'
appendix5_id = 'appendix5_data'
process_files_in_folder(folder_path, appendix2_id, appendix5_id)
