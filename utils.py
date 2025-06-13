"""
Utilities và helper functions cho hệ thống trích xuất từ khóa
"""

import json
from typing import Dict, List
from core_concept_extractor import SeedKeywords, ConceptMatrix


class KeywordAnalyzer:
    """Phân tích và đánh giá chất lượng từ khóa"""
    
    @staticmethod
    def analyze_keyword_quality(keywords: SeedKeywords) -> Dict[str, any]:
        """Phân tích chất lượng từ khóa"""
        analysis = {
            "total_keywords": 0,
            "category_distribution": {},
            "keyword_lengths": [],
            "quality_score": 0.0,
            "recommendations": []
        }
        
        all_keywords = []
        
        for category, keyword_list in keywords.dict().items():
            count = len(keyword_list)
            analysis["category_distribution"][category] = count
            analysis["total_keywords"] += count
            
            for keyword in keyword_list:
                all_keywords.append(keyword)
                analysis["keyword_lengths"].append(len(keyword.split()))
        
        # Tính điểm chất lượng
        if analysis["total_keywords"] > 0:
            avg_length = sum(analysis["keyword_lengths"]) / len(analysis["keyword_lengths"])
            balance_score = min(1.0, len([c for c in analysis["category_distribution"].values() if c > 0]) / 6)
            length_score = min(1.0, avg_length / 2)  # Ideal 2 words per keyword
            
            analysis["quality_score"] = (balance_score + length_score) / 2
        
        # Đề xuất cải thiện
        empty_categories = [cat for cat, count in analysis["category_distribution"].items() if count == 0]
        if empty_categories:
            analysis["recommendations"].append(f"Cần bổ sung từ khóa cho: {', '.join(empty_categories)}")
        
        if analysis["quality_score"] < 0.7:
            analysis["recommendations"].append("Chất lượng từ khóa cần cải thiện")
        
        return analysis
    
    @staticmethod
    def export_to_json(results: Dict, filename: str):
        """Xuất kết quả ra file JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Đã xuất kết quả ra file: {filename}")
    
    @staticmethod
    def format_for_search_engine(keywords: SeedKeywords) -> Dict[str, List[str]]:
        """Format từ khóa cho công cụ tìm kiếm sáng chế"""
        formatted = {
            "primary_keywords": [],
            "secondary_keywords": [],
            "technical_terms": [],
            "application_domains": []
        }
        
        # Primary: Problem + Object + Key Technical Features
        formatted["primary_keywords"].extend(keywords.problem_purpose)
        formatted["primary_keywords"].extend(keywords.object_system)
        formatted["primary_keywords"].extend(keywords.key_technical_feature)
        
        # Secondary: Action + Advantage
        formatted["secondary_keywords"].extend(keywords.action_method)
        formatted["secondary_keywords"].extend(keywords.advantage_result)
        
        # Technical terms: Key Technical Features
        formatted["technical_terms"] = keywords.key_technical_feature
        
        # Application domains: Environment/Field
        formatted["application_domains"] = keywords.environment_field
        
        return formatted


class PatentSearchQuery:
    """Tạo truy vấn tìm kiếm sáng chế từ từ khóa"""
    
    @staticmethod
    def create_boolean_query(keywords: SeedKeywords) -> str:
        """Tạo truy vấn Boolean cho database sáng chế"""
        primary = keywords.problem_purpose + keywords.object_system + keywords.key_technical_feature
        secondary = keywords.action_method + keywords.advantage_result
        
        query_parts = []
        
        if primary:
            primary_query = " OR ".join([f'"{kw}"' for kw in primary[:3]])  # Lấy 3 từ khóa quan trọng nhất
            query_parts.append(f"({primary_query})")
        
        if secondary:
            secondary_query = " OR ".join([f'"{kw}"' for kw in secondary[:2]])
            query_parts.append(f"({secondary_query})")
        
        if keywords.environment_field:
            field_query = " OR ".join([f'"{field}"' for field in keywords.environment_field])
            query_parts.append(f"AND ({field_query})")
        
        return " AND ".join(query_parts)
    
    @staticmethod
    def create_natural_query(keywords: SeedKeywords) -> str:
        """Tạo truy vấn ngôn ngữ tự nhiên"""
        all_keywords = []
        for keyword_list in keywords.dict().values():
            all_keywords.extend(keyword_list)
        
        # Chọn các từ khóa quan trọng nhất
        important_keywords = all_keywords[:8]  # Giới hạn 8 từ khóa
        
        return " ".join(important_keywords)


class ReportGenerator:
    """Tạo báo cáo chi tiết"""
    
    @staticmethod
    def generate_extraction_report(
        input_text: str, 
        concept_matrix: ConceptMatrix, 
        final_keywords: SeedKeywords,
        messages: List[str]
    ) -> str:
        """Tạo báo cáo chi tiết quá trình trích xuất"""
        
        analyzer = KeywordAnalyzer()
        analysis = analyzer.analyze_keyword_quality(final_keywords)
        search_query = PatentSearchQuery()
        
        report = f"""
# BÁO CÁO TRÍCH XUẤT TỪ KHÓA GỐC SÁNG CHẾ

## 📄 Nội dung đầu vào
{input_text[:500]}{'...' if len(input_text) > 500 else ''}

## 📋 Ma trận Khái niệm
- **Vấn đề/Mục tiêu**: {concept_matrix.problem_purpose}
- **Đối tượng/Hệ thống**: {concept_matrix.object_system}
- **Hành động/Phương pháp**: {concept_matrix.action_method}
- **Đặc điểm kỹ thuật**: {concept_matrix.key_technical_feature}
- **Môi trường/Lĩnh vực**: {concept_matrix.environment_field}
- **Lợi ích/Kết quả**: {concept_matrix.advantage_result}

## 🔑 Từ khóa gốc cuối cùng
"""
        
        for category, keywords in final_keywords.dict().items():
            category_name = category.replace('_', ' ').title()
            report += f"- **{category_name}**: {', '.join(keywords)}\n"
        
        report += f"""
## 📊 Phân tích chất lượng
- **Tổng số từ khóa**: {analysis['total_keywords']}
- **Điểm chất lượng**: {analysis['quality_score']:.2f}/1.0
- **Chiều dài trung bình**: {sum(analysis['keyword_lengths'])/len(analysis['keyword_lengths']):.1f} từ

### Phân bố theo danh mục:
"""
        
        for category, count in analysis['category_distribution'].items():
            category_name = category.replace('_', ' ').title()
            report += f"- {category_name}: {count} từ khóa\n"
        
        if analysis['recommendations']:
            report += "\n### 💡 Đề xuất cải thiện:\n"
            for rec in analysis['recommendations']:
                report += f"- {rec}\n"
        
        report += f"""
## 🔍 Truy vấn tìm kiếm đề xuất

### Boolean Query:
```
{search_query.create_boolean_query(final_keywords)}
```

### Natural Language Query:
```
{search_query.create_natural_query(final_keywords)}
```

## 📝 Lịch sử xử lý
"""
        
        for i, message in enumerate(messages, 1):
            report += f"{i}. {message}\n"
        
        return report


if __name__ == "__main__":
    # Test utilities
    print("🧪 Testing utilities...")
    
    # Sample data for testing
    sample_keywords = SeedKeywords(
        problem_purpose=["water conservation", "automatic irrigation"],
        object_system=["irrigation system"],
        action_method=["control", "schedule"],
        key_technical_feature=["soil moisture sensor", "weather data"],
        environment_field=["agriculture", "gardening"],
        advantage_result=["optimize water usage", "reduce cost"]
    )
    
    # Test analyzer
    analyzer = KeywordAnalyzer()
    analysis = analyzer.analyze_keyword_quality(sample_keywords)
    print("📊 Quality Analysis:")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
    
    # Test search query
    query_gen = PatentSearchQuery()
    boolean_query = query_gen.create_boolean_query(sample_keywords)
    print(f"\n🔍 Boolean Query: {boolean_query}")
    
    natural_query = query_gen.create_natural_query(sample_keywords)
    print(f"🔍 Natural Query: {natural_query}")
