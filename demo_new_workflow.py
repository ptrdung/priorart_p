#!/usr/bin/env python3
"""
Demo script cho workflow mới với reflection và human-in-loop
"""

from core_concept_extractor import CoreConceptExtractor
import json


def main():
    print("🚀 Demo Workflow Mới với Reflection và Human-in-Loop")
    print("="*70)
    
    # Khởi tạo extractor
    extractor = CoreConceptExtractor(model_name="qwen2.5:0.5b-instruct", use_checkpointer=False)
    
    # Văn bản mẫu về một hệ thống tưới tiêu thông minh
    sample_text = """
    Một hệ thống tưới tiêu thông minh sử dụng cảm biến độ ẩm đất và dữ liệu thời tiết
    để tự động điều khiển lịch trình tưới nước. Hệ thống giúp tiết kiệm nước và 
    tối ưu hóa việc chăm sóc cây trồng trong nông nghiệp và làm vườn.
    
    Hệ thống bao gồm:
    - Cảm biến độ ẩm đất capacitive
    - Module Wi-Fi ESP32 để kết nối internet
    - API thời tiết để lấy dự báo
    - Thuật toán machine learning để dự đoán nhu cầu nước
    - Van điện từ để điều khiển dòng nước
    - Ứng dụng di động để giám sát từ xa
    
    Ưu điểm:
    - Tiết kiệm nước lên đến 40%
    - Giảm thời gian chăm sóc cây trồng
    - Tăng năng suất cây trồng
    - Giám sát từ xa qua smartphone
    """
    
    print("📄 Văn bản đầu vào:")
    print("-" * 50)
    print(sample_text)
    print("\n" + "="*70)
    
    print("\n🔄 Bắt đầu workflow 4 bước:")
    print("B1: Tạo bản tóm tắt theo các field")
    print("B2: Tạo keyword chính cho các fields")  
    print("B3: Reflection đánh giá keywords")
    print("B4: Human in the loop")
    print("\n" + "="*70)
    
    try:
        # Chạy workflow
        results = extractor.extract_keywords(sample_text)
        
        print("\n" + "="*70)
        print("📊 KẾT QUẢ CUỐI CÙNG")
        print("="*70)
        
        if results["final_keywords"]:
            print("\n🔑 Keywords cuối cùng:")
            for field, keywords in results["final_keywords"].items():
                field_name = field.replace('_', ' ').title()
                print(f"  📌 {field_name}: {keywords}")
        
        if results["reflection_evaluation"]:
            reflection = results["reflection_evaluation"]
            print(f"\n🤖 Đánh giá Reflection:")
            print(f"  • Chất lượng tổng thể: {reflection['overall_quality']}")
            print(f"  • Số lần reflection: {results['reflection_iterations']}")
            if reflection['issues_found']:
                print(f"  • Vấn đề tìm thấy: {len(reflection['issues_found'])} vấn đề")
                for issue in reflection['issues_found'][:3]:
                    print(f"    - {issue}")
        
        print(f"\n📝 Hành động người dùng: {results.get('user_action', 'Không có')}")
        
        print("\n📋 Quá trình xử lý:")
        for i, msg in enumerate(results["messages"], 1):
            print(f"  {i}. {msg}")
            
        print("\n✅ Demo hoàn thành!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo bị dừng bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình demo: {e}")
        print("Đảm bảo Ollama đang chạy và model qwen2.5:0.5b-instruct khả dụng")


if __name__ == "__main__":
    main()
