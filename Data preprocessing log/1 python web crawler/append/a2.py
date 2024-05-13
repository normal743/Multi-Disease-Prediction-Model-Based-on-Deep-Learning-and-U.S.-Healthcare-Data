import requests
import pandas as pd
from bs4 import BeautifulSoup

def read_and_save_appendix_table(url, appendix_id, file_name):
    """
    从指定的 URL 中读取指定附录的表格数据，并保存为 CSV 文件。

    Parameters:
        url (str): 要读取的网页的 URL。
        appendix_id (str): 要读取的附录的 ID。
        file_name (str): 要保存的 CSV 文件名。

    Returns:
        DataFrame: 包含表格数据的 DataFrame。
    """
    # 发送 HTTP 请求并获取网页内容
    response = requests.get(url)

    # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # 找到指定附录的标题所在的标签
    appendix_title_tag = soup.find('h2', {'id': appendix_id})
    # 如果找到了标题标签，则找到下一个表格
    if appendix_title_tag:
        # 找到下一个表格
        appendix_table_tag = appendix_title_tag.find_next('table')
        # 如果找到表格，则提取其中的文本
        if appendix_table_tag:
            # 提取表格数据
            data = []
            rows = appendix_table_tag.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if cols:  # 如果列不为空
                    cols = [col.get_text() for col in cols]
                    data.append(cols)
            # 将表格数据转换为 DataFrame
            df = pd.DataFrame(data[1:], columns=data[0])  # 第一行为表头，其余为数据
            # 保存 DataFrame 到 CSV 文件
            df.to_csv(file_name, index=False)
            print(f"{file_name} 文件已保存成功")
            return df
        else:
            print(f"{appendix_id} 表格未找到")
    else:
        print(f"{appendix_id} 标题未找到")

# 示例使用方法
url = 'https://wwwn.cdc.gov/nchs/nhanes/2013-2014/DR1IFF_H.htm#Codebook'
appendix2_id = 'Appendix_2._Variables_in_the_Individual_Foods_Files_(DR1IFF_H_and_DR2IFF_H)_by_Position'
appendix2_file = 'appendix2_data.csv'
appendix5_id = 'Appendix_5._Variables_in_the_Total_Nutrients_Files_(DR1TOT_H_and_DR2TOT_H)_by_Position'
appendix5_file = 'appendix5_data.csv'

appendix2_data = read_and_save_appendix_table(url, appendix2_id, appendix2_file)
appendix5_data = read_and_save_appendix_table(url, appendix5_id, appendix5_file)
