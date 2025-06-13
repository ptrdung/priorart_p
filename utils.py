"""
Utilities and helper functions for keyword extraction system
"""

import json
from typing import Dict, List
from core_concept_extractor import SeedKeywords, ConceptMatrix


class KeywordAnalyzer:
    """Analyze and evaluate keyword quality"""
    
    @staticmethod
    def analyze_keyword_quality(keywords: SeedKeywords) -> Dict[str, any]:
        """Analyze keyword quality"""
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
        
        # Calculate quality score
        if analysis["total_keywords"] > 0:
            avg_length = sum(analysis["keyword_lengths"]) / len(analysis["keyword_lengths"])
            balance_score = min(1.0, len([c for c in analysis["category_distribution"].values() if c > 0]) / 6)
            length_score = min(1.0, avg_length / 2)  # Ideal 2 words per keyword
            
            analysis["quality_score"] = (balance_score + length_score) / 2
        
        # Improvement recommendations
        empty_categories = [cat for cat, count in analysis["category_distribution"].items() if count == 0]
        if empty_categories:
            analysis["recommendations"].append(f"Need to add keywords for: {', '.join(empty_categories)}")
        
        if analysis["quality_score"] < 0.7:
            analysis["recommendations"].append("Keyword quality needs improvement")
        
        # Additional recommendations for the new workflow
        if analysis["total_keywords"] < 6:
            analysis["recommendations"].append("Consider adding more keywords for better coverage")
        
        if any(len(keyword_list) == 0 for keyword_list in keywords.dict().values()):
            analysis["recommendations"].append("Some categories are missing keywords entirely")
        
        return analysis
    
    @staticmethod
    def export_to_json(results: Dict, filename: str):
        """Export results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results exported to file: {filename}")
    
    @staticmethod
    def format_for_search_engine(keywords: SeedKeywords) -> Dict[str, List[str]]:
        """Format keywords for patent search engines"""
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
    """Create patent search queries from keywords"""
    
    @staticmethod
    def create_boolean_query(keywords: SeedKeywords) -> str:
        """Create Boolean query for patent databases"""
        primary = keywords.problem_purpose + keywords.object_system + keywords.key_technical_feature
        secondary = keywords.action_method + keywords.advantage_result
        
        query_parts = []
        
        if primary:
            primary_query = " OR ".join([f'"{kw}"' for kw in primary[:3]])  # Take 3 most important keywords
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
        """Create natural language query"""
        all_keywords = []
        for keyword_list in keywords.dict().values():
            all_keywords.extend(keyword_list)
        
        # Select most important keywords
        important_keywords = all_keywords[:8]  # Limit to 8 keywords
        
        return " ".join(important_keywords)


class ReportGenerator:
    """Generate detailed reports"""
    
    @staticmethod
    def generate_extraction_report(
        input_text: str, 
        concept_matrix: ConceptMatrix, 
        final_keywords: SeedKeywords,
        messages: List[str]
    ) -> str:
        """Generate detailed extraction process report"""
        
        analyzer = KeywordAnalyzer()
        analysis = analyzer.analyze_keyword_quality(final_keywords)
        search_query = PatentSearchQuery()
        
        report = f"""
# PATENT SEED KEYWORD EXTRACTION REPORT

## 📄 Input Content
{input_text[:500]}{'...' if len(input_text) > 500 else ''}

## 📋 Concept Matrix
- **Problem/Purpose**: {concept_matrix.problem_purpose}
- **Object/System**: {concept_matrix.object_system}
- **Action/Method**: {concept_matrix.action_method}
- **Key Technical Feature**: {concept_matrix.key_technical_feature}
- **Environment/Field**: {concept_matrix.environment_field}
- **Advantage/Result**: {concept_matrix.advantage_result}

## 🔑 Final seed keywords
"""
        
        for category, keywords in final_keywords.dict().items():
            category_name = category.replace('_', ' ').title()
            report += f"- **{category_name}**: {', '.join(keywords)}\n"
        
        report += f"""
## 📊 Quality Analysis
- **Total keywords**: {analysis['total_keywords']}
- **Quality score**: {analysis['quality_score']:.2f}/1.0
- **Average length**: {sum(analysis['keyword_lengths'])/len(analysis['keyword_lengths']):.1f} words

### Category distribution:
"""
        
        for category, count in analysis['category_distribution'].items():
            category_name = category.replace('_', ' ').title()
            report += f"- {category_name}: {count} keywords\n"
        
        if analysis['recommendations']:
            report += "\n### 💡 Improvement recommendations:\n"
            for rec in analysis['recommendations']:
                report += f"- {rec}\n"
        
        report += f"""
## 🔍 Recommended search queries

### Boolean Query:
```
{search_query.create_boolean_query(final_keywords)}
```

### Natural Language Query:
```
{search_query.create_natural_query(final_keywords)}
```

## 📝 Processing history
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
