import json

with open('D:/Python/Talang.in/testing/synthetic data/talangin_synthetic_templates_2.json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)

with open('D:/Python/Talang.in/testing/synthetic data/talangin_synthetic_templates.json', 'r', encoding='utf-8') as f:
    data2 = json.load(f)

combined_data = data1 + data2

with open('D:/Python/Talang.in/testing/synthetic data/templates_final.json', 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
print(f"Successfully joined {len(combined_data)} items!")