**Verdict**

Ở mức idea: **strong borderline / weak accept potential**, nhưng ở mức paper AAAI Main Track hiện tại: **borderline reject nếu chưa có kết quả thực nghiệm rất sạch**. Ý tưởng đã có “spine” tốt hơn bản dàn trải: một substrate `Fact Ledger`, một retriever disentangled, một training scheme bằng negative có cấu trúc. Điểm sáng nhất là liên kết C1–C2 ở [core_method.md](</Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Xử lý ngôn ngữ tự nhiên/NLP/core_method.md:22>): negative sinh từ ledger và huấn luyện đúng kênh scoring.

**Reviewer View**

Điểm mạnh lớn nhất: bài có một failure mode rõ của dense retrieval trong financial QA: entity, time, metric, value bị chìm trong vector tổng. Cách tách `concept / magnitude / entity-time` ở [line 42](</Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Xử lý ngôn ngữ tự nhiên/NLP/core_method.md:42>) và scoring ở [line 76](</Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Xử lý ngôn ngữ tự nhiên/NLP/core_method.md:76>) là hợp lý, dễ hiểu, đúng domain.

Nhưng claim **“provably-true”** ở [line 115](</Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Xử lý ngôn ngữ tự nhiên/NLP/core_method.md:115>) đang quá mạnh. Với query so sánh, tổng hợp, ratio, hoặc câu hỏi có nhiều fact thay thế, việc swap một fact không luôn làm document “chắc chắn không liên quan”. Nên đổi thành: **answer-invalidating negatives under explicit ledger-answer assumptions**. AAAI reviewer sẽ bắt lỗi chữ “provably” rất nhanh.

Điểm rủi ro thứ hai là **novelty có thể bị xem là tổ hợp kỹ thuật đã biết**: late interaction kiểu ColBERT, quantity-aware retrieval kiểu Numbers Matter, metadata filtering/filtered ANN, synthetic negatives. Contribution cần được đóng đinh là: **ledger-grounded, channel-aligned hard negatives for fact-level financial retrieval**, không phải “masked HNSW” riêng lẻ.

Điểm rủi ro thứ ba là extraction. Bạn đã nhận diện đúng ở [line 93](</Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Xử lý ngôn ngữ tự nhiên/NLP/core_method.md:93>), nhưng paper sẽ sống chết ở **answer-fact extraction recall**, không chỉ F1 tổng. Nếu ledger miss đúng cell trả lời, toàn bộ phương pháp mất lợi thế.

**AC / Editor View**

Tôi sẽ yêu cầu rebuttal cho 4 câu:

1. LEDGER thắng bao nhiêu so với **dense + hard metadata filter + reranker** ở [line 138](</Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Xử lý ngôn ngữ tự nhiên/NLP/core_method.md:138>)?
2. Synthetic negatives có tạo artifact dễ học không?
3. Filtered ANN có giữ recall khi mask rất nhỏ không?
4. Auto-ledger gap so với oracle-ledger là bao nhiêu?

Nếu bạn trả lời được bằng ablation tốt, bài có cửa. Nếu chỉ có story đẹp, chưa đủ AAAI.

**Edits Nên Làm**

- [Line 10](</Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Xử lý ngôn ngữ tự nhiên/NLP/core_method.md:10>): đổi “Chưa phương pháp nào...” thành “To our knowledge...” và tránh tuyệt đối hóa.
- [Line 50](</Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Xử lý ngôn ngữ tự nhiên/NLP/core_method.md:50>): đổi “nhất thiết thua” thành “khi nào single-vector thất bại”.
- [Line 63](</Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Xử lý ngôn ngữ tự nhiên/NLP/core_method.md:63>): “không thể bị nhấn” cũng hơi quá; nên viết “less likely to be dominated”.
- [Line 111](</Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Xử lý ngôn ngữ tự nhiên/NLP/core_method.md:111>): loss không chứa `d^-_v` dù trước đó nói 5 loại negative. Cần sửa consistency.
- [Line 136](</Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Xử lý ngôn ngữ tự nhiên/NLP/core_method.md:136>): kiểm tra lại số context. T²-RAGBench official ghi **7,318** reports, không phải 7,317.
- [Line 129](</Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Xử lý ngôn ngữ tự nhiên/NLP/core_method.md:129>): phần generation/verifier nên để appendix hoặc “downstream analysis”; đừng để reviewer nghĩ bài ôm thêm RAG generation.

**AAAI Chair Take**

Tôi sẽ pitch bài này như sau: **“Financial retrieval fails because relevance is fact-indexed, not document-indexed; LEDGER makes fact keys first-class at both inference and training.”** Đừng pitch là một architecture nhiều module. Pitch là một principle.

Nguồn mình đã đối chiếu nhanh: [T²-RAGBench](https://t2ragbench.demo.hcds.uni-hamburg.de/), [Numbers Matter!](https://aclanthology.org/2024.findings-emnlp.707/), [Granularity Dilemma](https://aclanthology.org/2025.findings-emnlp.1051.pdf), [Numeracy Gap](https://aclanthology.org/2026.findings-eacl.289/).