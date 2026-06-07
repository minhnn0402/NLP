# Đề cương Nghiên cứu: Kiến trúc RAG Nhất quán Ràng buộc Miền cho Bài toán Hỏi-Đáp Tài chính

## Tóm tắt Ý tưởng

Các hệ thống Hỏi-Đáp (QA) tự động trên tài liệu tài chính đòi hỏi khả năng suy luận chính xác trên các bảng biểu và văn bản, đồng thời phải tuân thủ nghiêm ngặt các định luật kế toán. Các kiến trúc Truy hồi-Tăng cường Sinh (RAG) hiện nay chủ yếu dựa trên tương đồng ngữ nghĩa thuần túy, thường thiếu cơ chế kiểm chứng các ràng buộc số học và logic miền, dẫn đến sai sót trong cả hai giai đoạn truy hồi và sinh câu trả lời. Nghiên cứu này đề xuất một kiến trúc RAG mới, trong đó toàn bộ pipeline được chi phối bởi một Đồ thị Tri thức Kế toán (Accounting Knowledge Graph). Đồ thị này mã hóa các đồng nhất thức kế toán phổ quát và đóng vai trò là tín hiệu ràng buộc xuyên suốt:

- **Pha 1 – Truy hồi có Cấu trúc Đồ thị (Graph-Structured Retrieval - GSR):** Tích hợp tương đồng văn bản, đối sánh thực thể, và điểm số nhất quán kế toán để truy hồi chính xác bằng chứng, đặc biệt trong các trường hợp tồn tại "nhiễu cứng" (hard negative) – những tài liệu có độ tương đồng từ vựng cao nhưng sai lệch về cấu trúc tài chính.
- **Pha 2 – Tinh chỉnh Nhận thức Ràng buộc (Constraint-Aware Fine-Tuning - CAFT):** Tinh chỉnh Mô hình Ngôn ngữ Lớn (LLM) để sinh câu trả lời nhất quán với các định luật kế toán. Phương pháp này sử dụng Tối ưu hóa Sở thích Trực tiếp (DPO), trong đó các cặp ưu tiên được thiết lập dựa trên một hàm thưởng kép, tách bạch giữa *độ chính xác* và *mức độ vi phạm ràng buộc*. Việc lượng giá mức độ vi phạm được thực hiện ngoài đồ thị tính toán, đảm bảo tính khả vi của toàn bộ quá trình huấn luyện.

Tính nhất quán của kiến trúc được củng cố bởi việc tái sử dụng cùng một thư viện mẫu kế toán và cùng một hàm điểm ràng buộc cho cả hai pha. Ngoài ra, một bộ dữ liệu tài chính tiếng Việt được xây dựng nhằm đánh giá khả năng chuyển giao ngôn ngữ của phương pháp đề xuất.

## 1. Giới thiệu

Hỏi-đáp trên tài liệu tài chính đặt ra những thách thức đặc thù vượt ra ngoài phạm vi của các hệ thống QA văn bản thuần túy. Các câu hỏi thường yêu cầu truy xuất và suy luận trên các bảng biểu phức tạp, đối chiếu số liệu giữa các kỳ kế toán, và quan trọng nhất, mọi suy luận phải tuân thủ các đồng nhất thức kế toán căn bản (ví dụ: Tổng tài sản = Tổng nợ phải trả + Vốn chủ sở hữu). Các kiến trúc RAG hiện đại thường bộc lộ hai hạn chế trong bối cảnh này:

1.  **Truy hồi thiếu hiểu biết miền:** Các bộ truy hồi chỉ dựa trên ngữ nghĩa thuần túy dễ bị đánh lừa bởi các bảng biểu có từ vựng tương tự nhưng khác biệt về thực thể (công ty, năm) hoặc về cấu trúc logic kế toán.
2.  **Sinh thiếu kiểm chứng:** Các LLM có thể sinh ra các phát biểu số học trôi chảy về mặt ngôn ngữ nhưng vi phạm các định luật kế toán, một dạng thức của "ảo giác" logic đặc thù của miền tài chính.

Để giải quyết các thách thức trên, nghiên cứu này đề xuất **Domain-Constraint RAG**, một kiến trúc lấy *ràng buộc miền* làm trung tâm, tích hợp chúng như một tín hiệu học thống nhất từ truy hồi đến sinh.

## 2. Đóng góp Dự kiến

Nghiên cứu dự kiến có các đóng góp chính sau:

1.  **Bộ dữ liệu Tài chính Tiếng Việt cho RAG:** Công bố một tập dữ liệu hỏi-đáp tài chính tiếng Việt có cấu trúc, bao gồm câu hỏi, ngữ cảnh văn bản – bảng biểu, metadata doanh nghiệp, và đáp án nền. Tập dữ liệu này phản ánh đặc thù của Chuẩn mực Kế toán Việt Nam (VAS) và phục vụ như một phép thử cho khả năng chuyển giao ngôn ngữ.
2.  **Graph-Structured Retrieval (GSR):** Đề xuất một bộ truy hồi mới kết hợp embedding văn bản, đối sánh thực thể, và *điểm số nhất quán đồ thị* được tính từ Đồ thị Tri thức Kế toán. GSR được chứng minh là đạt hiệu năng cạnh tranh trên các benchmark tiếng Anh (FinQA, ConvFinQA, TAT-DQA) và thích ứng nhanh với tiếng Việt.
3.  **Constraint-Aware Fine-Tuning (CAFT):** Đề xuất một phương pháp tinh chỉnh LLM dựa trên sở thích (DPO), trong đó các cặp so sánh được tạo ra dựa trên mức độ *vi phạm ràng buộc kế toán* của câu trả lời. CAFT cho phép LLM nội tại hóa các ràng buộc này mà không yêu cầu can thiệp vào kiến trúc sinh trong quá trình suy luận.
4.  **Pipeline nhất quán Thần kinh – Biểu tượng (Neuro-Symbolic):** Lần đầu tiên, cùng một cơ sở tri thức biểu tượng (KG) và cùng một hàm phạt ràng buộc được sử dụng xuyên suốt để huấn luyện cả bộ truy hồi (thông qua học tương phản) lẫn bộ sinh (thông qua tối ưu hóa sở thích), tạo nên một kiến trúc RAG đồng bộ và chặt chẽ.

## 3. Kiến trúc Tổng thể

Pipeline bao gồm hai pha chính, được thiết kế để vận hành đồng bộ.

### Pha 1: Truy hồi có Cấu trúc Đồ thị (GSR)

#### a. Xây dựng Đồ thị Tri thức từ Bảng Tài chính
Mỗi tài liệu chứa bảng được phân tích và biểu diễn thành một đồ thị:
- Trích xuất các ô dữ liệu làm nút, mang giá trị số, nhãn hàng/cột.
- Chuẩn hóa nhãn và khớp với một thư viện gồm 15+ mẫu template kế toán (được xây dựng dựa trên IFRS/GAAP/VAS).
- Khi khớp thành công, các cạnh kế toán (accounting edge) có trọng số $\omega = +1$ (cho quan hệ cộng dồn) hoặc $-1$ (cho quan hệ trừ hao) được thiết lập. Trong trường hợp không khớp, một cạnh vị trí (positional edge) với $\omega = 0$ được sử dụng như một cơ chế dự phòng.

Đầu ra là một đồ thị $G_D$ mã hóa các ràng buộc kế toán của tài liệu đó.

#### b. Mã hóa và Truy hồi
- Mỗi đồ thị $G_D$ được mã hóa thành vector embedding $\mathbf{h}_G$ bằng Mạng Chú ý Đồ thị (GAT) có nhận thức về trọng số cạnh.
- Một chỉ mục FAISS được xây dựng trên các embedding văn bản đa ngữ của tài liệu (sử dụng BGE-M3, multilingual-e5, v.v.) để truy hồi thô.
- Điểm kết hợp (joint score) cho mỗi cặp câu hỏi-tài liệu được tính như sau:
  $$s(Q, D) = \alpha \cdot s_{\text{text}} + \beta \cdot s_{\text{entity}} + \gamma \cdot s_{\text{constraint}}(G_D)$$
  trong đó $s_{\text{constraint}}(G_D) = \frac{1}{|E_{\text{acc}}|}\sum_{e \in E_{\text{acc}}} \exp\left(-\frac{|\omega_e \cdot v_u - v_v|}{\max(|v_v|,\epsilon)}\right)$, đại diện cho mức độ nhất quán của bảng với các đồng nhất thức kế toán.

#### c. Huấn luyện với Học Tương phản Nhận thức Ràng buộc (CACL)
Bộ truy hồi được huấn luyện với một bộ sinh nhiễu cứng chuyên biệt (CHAP), tạo ra ba loại mẫu âm:
- **CHAP-A:** Sửa đổi một giá trị ô để phá vỡ đồng nhất thức cộng/trừ.
- **CHAP-S:** Thay đổi thang đo (ví dụ: triệu đồng thành tỷ đồng), làm sai lệch quan hệ số học.
- **CHAP-E:** Thay đổi thực thể (công ty/năm) để tạo mẫu âm về mặt metadata.

Hàm mất mát huấn luyện là sự kết hợp của mất mát bộ ba (triplet loss) và một thành phần phạt ràng buộc: $\mathcal{L}_{\text{CACL}} = \mathcal{L}_{\text{triplet}} + \lambda \cdot \mathcal{L}_{\text{constraint}}$. Mục tiêu là đảm bảo các tài liệu dương có điểm số cao hơn hẳn các mẫu âm, đồng thời trừng phạt nặng các mẫu âm vi phạm ràng buộc bị gán điểm cao.

### Pha 2: Sinh Nhận thức Ràng buộc (CAFT)

Pha này mở rộng ý tưởng ràng buộc sang khâu sinh câu trả lời. Thay vì đưa điểm ràng buộc vào như một số hạng trong hàm mất mát (gây ra vấn đề không khả vi do yêu cầu phải phân tích cú pháp đầu ra văn bản), chúng tôi chuyển hóa nó thành tín hiệu ưu tiên cho Tối ưu hóa Sở thích Trực tiếp (DPO).

#### a. Tạo dữ liệu ưu tiên (Preference Data)
1.  **Prompt có cấu trúc:** LLM được hướng dẫn sinh lập luận từng bước và kết thúc bằng một dòng chuẩn hóa `ANSWER: <tên_biến>=<giá_trị>;...` nhằm hỗ trợ việc trích xuất các biến số một cách đáng tin cậy.
2.  **Sinh tập ứng viên (Rollout):** Với mỗi câu hỏi, một tập các câu trả lời được sinh ra từ mô hình cơ sở bằng các chiến lược giải mã đa dạng.
3.  **Tính hàm thưởng kép:**
    *   $\mathcal{R}_{\text{acc}}$: Đo lường độ chính xác của đáp án cuối cùng so với nền mẫu (ground truth).
    *   $\mathcal{R}_{\text{cstr}}$: Trích xuất các biến từ dòng `ANSWER:`, đối chiếu với các ràng buộc $c \in \mathcal{C}$ từ đồ thị $G_D$ của tài liệu đã truy hồi. Mức độ vi phạm $v_c$ và điểm thưởng được tính như sau:
        $$v_c = \frac{|\text{LHS}(V_{\text{pred}}) - \text{RHS}(V_{\text{pred}})|}{\max(|\text{LHS}(V_{\text{pred}})|, \epsilon)}$$
        $$\mathcal{R}_{\text{cstr}} = \frac{1}{|\mathcal{C}|}\sum_{c} \exp(-v_c)$$
    *   $\mathcal{R}_{\text{total}} = \alpha \mathcal{R}_{\text{acc}} + \beta \mathcal{R}_{\text{cstr}}$.
4.  **Xây dựng cặp ưu tiên $(y_{\text{win}}, y_{\text{lose}})$:** Từ tập ứng viên, câu trả lời có $\mathcal{R}_{\text{total}}$ cao nhất được chọn làm $y_{\text{win}}$ và câu có $\mathcal{R}_{\text{total}}$ thấp nhất làm $y_{\text{lose}}$. Chiến lược này đặc biệt ưu tiên các câu trả lời nhất quán về mặt kế toán, ngay cả khi chúng có sai số nhỏ về giá trị trích xuất, hơn là các câu trả lời vô tình đúng số nhưng lập luận vi phạm ràng buộc.

#### b. Tinh chỉnh bằng DPO
Mô hình LLM được tinh chỉnh để tối thiểu hóa mất mát DPO trên các cặp đã xây dựng: $\mathcal{L}_{\text{CAFT}} = \mathcal{L}_{\text{DPO}}(\pi_\theta; y_{\text{win}}, y_{\text{lose}})$. Quá trình này hoàn toàn khả vi vì chỉ thao tác trên xác suất log của các chuỗi token, nhưng định hướng mô hình học cách ưu tiên các suy luận nhất quán với đồ thị tri thức.

#### c. Phân tích Kỹ thuật
Bước tính $\mathcal{R}_{\text{cstr}}$ (bao gồm phân tích cú pháp văn bản thành số) được thực hiện hoàn toàn ngoài đồ thị tính toán và chỉ phục vụ mục đích gán nhãn cho các cặp ưu tiên. Nhờ đó, gradient từ hàm DPO được lan truyền một cách trơn tru qua mô hình mà không gặp trở ngại từ phép biến đổi rời rạc text-to-float.

## 4. Kế hoạch Thực nghiệm

### 4.1. Thiết lập
- **Dữ liệu Tiếng Anh:** Các benchmark FinQA, ConvFinQA, và TAT-DQA cho cả hai bài toán truy hồi và sinh.
- **Dữ liệu Tiếng Việt:** Tập dữ liệu tự xây dựng gồm các câu hỏi đa dạng (truy xuất, tính toán, so sánh) trên các báo cáo tài chính thực tế.
- **Các mô hình nền:**
    - *Truy hồi:* BM25, Dense Retriever (BGE-M3, ColBERTv2), DTR.
    - *Sinh:* LLM không tinh chỉnh (Few-shot), RAG-TextOnly, RAG-GSR không dùng tín hiệu ràng buộc khi tinh chỉnh.
- **Độ đo:**
    - *Truy hồi:* MRR@3, Recall@3, NDCG@3.
    - *Sinh:* Exact Match (EM), Numerical Accuracy (NumAcc), Constraint Violation Rate (CVR) – tỉ lệ câu trả lời vi phạm ít nhất một ràng buộc kế toán.

### 4.2. Kết quả Kỳ vọng
- GSR đạt hiệu năng vượt trội so với các bộ truy hồi mạnh trên mọi benchmark, đặc biệt là trên các câu hỏi yêu cầu phân biệt các bảng có cấu trúc tương tự.
- CAFT làm giảm đáng kể CVR trong khi vẫn duy trì hoặc cải thiện EM và NumAcc so với các phương pháp tinh chỉnh không có nhận thức về ràng buộc.
- Trên tập dữ liệu tiếng Việt, pipeline chứng tỏ khả năng thích ứng hiệu quả với chi phí bản địa hóa thấp.

### 4.3. Nghiên cứu Triệt tiêu (Ablation Study)
Một nghiên cứu triệt tiêu toàn diện sẽ được thực hiện để làm rõ vai trò của từng thành phần:
- Phân tích mức độ ảnh hưởng của các trọng số $\alpha, \beta, \gamma$ trong GSR.
- So sánh hiệu năng của DPO (CAFT) với phương pháp SFT cộng phạt trực tiếp, qua đó minh họa cho tính ưu việt của cơ chế ưu tiên trong việc xử lý vấn đề không khả vi.
- Đánh giá tác động của tỉ lệ $\mathcal{R}_{\text{acc}}:\mathcal{R}_{\text{cstr}}$ trong việc hình thành cặp ưu tiên đến chất lượng sinh cuối cùng.

## 5. Kết luận

Nghiên cứu này giới thiệu một kiến trúc RAG nhất quán cho lĩnh vực tài chính, nơi tri thức miền kế toán được mã hóa thành một đồ thị và đóng vai trò là tín hiệu ràng buộc xuyên suốt cả hai pha truy hồi và sinh. Thông qua việc chuyển đổi điểm ràng buộc thành tín hiệu ưu tiên cho DPO, chúng tôi giải quyết một cách hiệu quả bài toán "làm sao để LLM tôn trọng các định luật kế toán" một cách khả vi và chặt chẽ. Các thực nghiệm trên cả tiếng Anh và tiếng Việt được kỳ vọng sẽ chứng minh tính ưu việt và khả năng tổng quát hóa của phương pháp được đề xuất.