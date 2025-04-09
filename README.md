# ChildPioneer｜兒童早療遊戲式肢體訓練

## 🧠 專案簡介
此專案為銘傳大學 113 年資訊管理學系畢業專題，目標透過遊戲式介面設計，協助兒童進行早期療育的肢體訓練。專案包括用戶認證、註冊、密碼重置、個人資料更新、背景圖片購買和選擇、治療安排和訓練模式選擇等功能。

## 🎯 專案目標與功能
- 目標：提供一個遊戲化的互動平台，幫助兒童進行肢體訓練，促進其早期療育。
- 主要功能：
    - 用戶註冊與登入
    - 密碼重置功能
    - 個人資料更新
    - 背景圖片購買與選擇
    - 治療安排
    - 訓練模式選擇
---

## 🔧 環境需求
- 最低 Python 版本需求：**3.9.12**
> 請至 [Python 官方網站](https://www.python.org/downloads/release/python-3912/) 下載並安裝對應版本。

---

## ⚙️ 安裝與執行步驟

### 1️⃣ Clone 專案
```bash
git clone https://github.com/yChiNL/ChildPioneer.git
``` 

### 2️⃣ 建立虛擬環境（venv）(僅第一次需要)
```bash
# 建立虛擬環境
python -m venv .venv
```

### 3️⃣ 啟動虛擬環境
```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4️⃣ 安裝所需插件
```bash
pip install -r requirements.txt
```

### 5️⃣ 啟動專案伺服器
```bash
python manage.py runserver
```

啟動後請在瀏覽器開啟：http://127.0.0.1:8000

每次啟動專案皆須執行步驟3️⃣&5️⃣

## 🧰 技術與工具
![Django](https://img.shields.io/badge/Django-092E20?style=flat&logo=django&logoColor=white) ![Python](https://img.shields.io/badge/Python-3.9-blue) ![SQLite3](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=SQLite&logoColor=white) ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-007ACC?logo=visualstudiocode&logoColor=fff&style=plastic) ![Git](https://img.shields.io/badge/-Git-F05032?style=flat-square&logo=git&logoColor=white) ![Pycharm](https://img.shields.io/badge/-Pycharm-3776AB?style=flat&logo=Pycharm&logoColor=white)
