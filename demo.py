"""
Demo script cho hệ thống trích xuất từ khóa gốc sáng chế
"""

from core_concept_extractor import CoreConceptExtractor
import json


def demo_with_sample_patents():
    """Demo với các mẫu sáng chế khác nhau"""
    
    extractor = CoreConceptExtractor(model_name="llama3")
    
    samples = [
        {
            "title": "Hệ thống tưới tiêu thông minh",
            "text": """
            Hệ thống tưới tiêu thông minh sử dụng cảm biến độ ẩm đất và dữ liệu thời tiết 
            để tự động điều khiển lịch tưới nước. Hệ thống bao gồm các cảm biến IoT, 
            bộ điều khiển trung tâm và ứng dụng di động. Giúp tiết kiệm nước lên đến 30% 
            và tối ưu hóa việc chăm sóc cây trồng trong nông nghiệp và làm vườn.
            """
        },
        {
            "title": "Robot dọn dẹp tự động",
            "text": """
            Robot dọn dẹp sử dụng công nghệ LIDAR và AI để lập bản đồ không gian,
            tránh vật cản và dọn dẹp hiệu quả. Robot có khả năng hút bụi, lau nhà
            và tự động trở về trạm sạc. Ứng dụng trong gia đình và văn phòng,
            giảm 80% thời gian dọn dẹp thủ công.
            """
        }
    ]
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*80}")
        print(f"🔬 DEMO {i}: {sample['title']}")
        print(f"{'='*80}")
        
        results = extractor.extract_keywords(sample['text'])
        
        print(f"\n📋 Kết quả cho '{sample['title']}':")
        print(json.dumps(results, indent=2, ensure_ascii=False))


def interactive_mode():
    """Chế độ tương tác cho người dùng nhập liệu"""
    
    extractor = CoreConceptExtractor(model_name="llama3")
    
    print("🚀 CHÀO MỪNG ĐÁN HỆ THỐNG TRÍCH XUẤT TỪ KHÓA GỐC SÁNG CHẾ")
    print("="*70)
    
    while True:
        print("\nVui lòng nhập mô tả ý tưởng hoặc tài liệu kỹ thuật:")
        print("(Nhập 'quit' để thoát)")
        
        user_input = input("\n📝 Nội dung: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'thoát']:
            print("👋 Cảm ơn bạn đã sử dụng hệ thống!")
            break
        
        if not user_input:
            print("⚠️ Vui lòng nhập nội dung hợp lệ!")
            continue
        
        try:
            print("\n🔄 Đang xử lý...")
            results = extractor.extract_keywords(user_input)
            
            print("\n" + "="*60)
            print("📊 KẾT QUẢ TRÍCH XUẤT")
            print("="*60)
            
            if results['final_keywords']:
                print("\n🔑 Từ khóa gốc cuối cùng:")
                for category, keywords in results['final_keywords'].items():
                    category_name = category.replace('_', ' ').title()
                    print(f"  • {category_name}: {keywords}")
            
            print(f"\n📝 Lịch sử xử lý:")
            for msg in results['messages']:
                print(f"  → {msg}")
                
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")
            print("Vui lòng thử lại với nội dung khác.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_with_sample_patents()
    else:
        interactive_mode()
