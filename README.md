怎麼建立虛擬環境:

1.  建立虛擬環境
```bash
python -m venv venv
```

2.  啟動虛擬環境
```bash
.\venv\Scripts\activate
```

3.  下載套件列表
```bash
pip install -r requirements.txt
```

4.  下載套件 (例: ultralytics)
```bash
pip install ultralytics
```

5. 更新套件列表
```bash
pip freeze > requirements.txt
```