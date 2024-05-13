import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

# 准备要爬取的网站列表
websites = [

    "https://wwwn.cdc.gov/nchs/nhanes/search/variablelist.aspx?Component=Dietary&CycleBeginYear=2005",
    "https://wwwn.cdc.gov/nchs/nhanes/search/variablelist.aspx?Component=Examination&CycleBeginYear=2005",
    "https://wwwn.cdc.gov/nchs/nhanes/search/variablelist.aspx?Component=Laboratory&CycleBeginYear=2005",
    # 添加第四个网站的链接
    "https://wwwn.cdc.gov/nchs/nhanes/search/variablelist.aspx?Component=Demographics&CycleBeginYear=2005",
    "https://wwwn.cdc.gov/Nchs/Nhanes/Search/variablelist.aspx?Component=Questionnaire&Cycle=2005-2006"
]

# 创建一个空的DataFrame来存储数据
combined_df = pd.DataFrame()

# 遍历每个网站并爬取数据
for website in tqdm(websites, desc="Scraping Websites"):
    # 发送GET请求获取网页内容
    response = requests.get(website)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 找到目标表格
    table = soup.find('table', {'id': 'GridView1'})
    
    # 提取表格数据
    data = []
    for tr in table.find_all('tr')[1:]:
        row = []
        for td in tr.find_all('td'):
            row.append(td.text.strip())
        data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=["Variable Name", "Variable Description", "Data File Name", 
                                     "Data File Description", "Begin Year", "EndYear", "Component", 
                                     "Use Constraints"])
    
    # 将当前DataFrame添加到组合DataFrame中
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# 将组合DataFrame保存为CSV文件
csv_file_path = 'com_data.csv'
combined_df.to_csv(csv_file_path, index=False)

# 打印CSV文件的地址
print("保存的CSV文件地址:", os.path.abspath(csv_file_path))
