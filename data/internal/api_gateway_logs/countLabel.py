import pandas as pd

INPUTFILE = r"n:/hodaoty/API-Threat-Detection/data/internal/api_gateway_logs/api_gateway_cleaned.csv"

# Đọc dữ liệu
df = pd.read_csv(INPUTFILE)

# Đếm số lượng theo giá trị của cột 'label'
label_counts = df['label'].value_counts()

print("Số lượng label 0:", label_counts.get(0, 0))
print("Số lượng label 1:", label_counts.get(1, 0))
