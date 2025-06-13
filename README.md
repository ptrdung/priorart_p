# Hệ thống Trích xuất Từ khóa Gốc Sáng chế

Hệ thống triển khai phương pháp 3 pha để trích xuất từ khóa gốc từ tài liệu kỹ thuật/sáng chế sử dụng LangChain, LangGraph và Ollama.

## 🏗️ Kiến trúc

- **LangChain**: Tích hợp LLM và xử lý prompt
- **LangGraph**: Workflow với human-in-the-loop
- **Ollama**: Local LLM (Llama3)
- **Pydantic**: Data validation và structure

## 📦 Cài đặt

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt Ollama (nếu chưa có)
# https://ollama.ai/

# Pull model Llama3
ollama pull llama3
```

## 🚀 Sử dụng

### 1. Chế độ tương tác

```bash
python demo.py
```

### 2. Chạy demo với mẫu có sẵn

```bash
python demo.py demo
```

### 3. Sử dụng programmatically

```python
from core_concept_extractor import CoreConceptExtractor

extractor = CoreConceptExtractor(model_name="llama3")
results = extractor.extract_keywords("Mô tả sáng chế của bạn...")
```

## 📋 Quy trình 3 pha

### Pha 1: Trừu tượng hóa & Định nghĩa Khái niệm

- Phân tích tài liệu đầu vào
- Tạo Ma trận Khái niệm với 6 thành phần

### Pha 2: Trích xuất Từ khóa Gốc

- Từ Ma trận Khái niệm → 1-3 từ khóa/thành phần
- Ưu tiên danh từ kỹ thuật và động từ chính

### Pha 3: Kiểm tra & Tinh chỉnh

- Human-in-the-loop validation
- Cải thiện dựa trên feedback người dùng

## 📁 Cấu trúc File

```text
priorart_project/
├── requirements.txt          # Dependencies
├── core_concept_extractor.py # Core system logic
├── demo.py                   # Demo và interactive mode  
├── utils.py                  # Utilities và analysis tools
└── README.md                 # Documentation
```

## 🔧 Tính năng

- ✅ Workflow 3 pha tự động
- ✅ Human-in-the-loop validation
- ✅ Phân tích chất lượng từ khóa
- ✅ Tạo truy vấn tìm kiếm Boolean/Natural Language
- ✅ Báo cáo chi tiết quá trình
- ✅ Export JSON results

## 🎯 Output

Hệ thống tạo ra:

1. **Ma trận Khái niệm**: 6 thành phần cốt lõi
2. **Từ khóa gốc**: 1-3 từ khóa/thành phần  
3. **Truy vấn tìm kiếm**: Boolean và Natural Language
4. **Báo cáo chất lượng**: Phân tích và đề xuất

## 📊 Example Output

```json
{
  "final_keywords": {
    "problem_purpose": ["water conservation", "automatic irrigation"],
    "object_system": ["irrigation system"],
    "action_method": ["control", "schedule"],
    "key_technical_feature": ["soil moisture sensor", "weather data"],
    "environment_field": ["agriculture", "gardening"],
    "advantage_result": ["optimize water usage", "reduce cost"]
  }
}
```

## 🔍 Search Queries

**Boolean Query:**

```text
("water conservation" OR "automatic irrigation" OR "irrigation system") AND ("control" OR "schedule") AND ("agriculture" OR "gardening")
```

**Natural Language Query:**

```text
water conservation automatic irrigation irrigation system control schedule soil moisture sensor weather data
```
