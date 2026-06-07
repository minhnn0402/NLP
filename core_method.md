# Kiến trúc Phương pháp: Constraint-Consistent Financial RAG

## Tổng quan Pipeline

Kiến trúc bao gồm hai pha chính, được tích hợp xuyên suốt bởi một **Thư viện Ràng buộc Kế toán** (Accounting Constraint Library) – một tập hợp các phương trình kế toán biểu tượng có thể tái sử dụng. Thư viện này cung cấp các **hàm đánh giá phương trình** (Equation Evaluators) được dùng nhất quán trong bốn vai trò: (i) tín hiệu xếp hạng lại phụ thuộc truy vấn trong truy hồi, (ii) quy tắc sinh nhiễu cứng cho huấn luyện tương phản, (iii) tín hiệu ưu tiên phân hạng cho tinh chỉnh mô hình sinh, và (iv) bộ kiểm chứng lúc suy luận để tự sửa lỗi.

```
Input Query
   │
   ▼
[GSR Retrieval]
   │ - Text + Entity + Query-Conditioned Constraint Score
   │ - Graph Construction with Equation-Level Constraint Nodes
   ▼
Top-k Evidence + Constraint Graphs (G_D)
   │
   ▼
[CAFT Generator]
   │
   ├─► Grounding Module (Evidence Selector, Cell Linker, Variable Canonicalizer, Constraint Selector)
   │
   ├─► Generation (LLM với cấu trúc ANSWER)
   │
   └─► Constraint Verifier (Equation Evaluators)
         │
         ├─ Nếu vi phạm → Self-Correction Prompt → Quay lại Generation
         │
         └─ Nếu đạt → Final Answer
```

---

## Pha 1: Graph-Structured Retrieval (GSR)

### 1.1 Xây dựng Đồ thị Tài liệu với Ràng buộc Cấp Phương trình

**Đầu vào**: Một tài liệu chứa bảng tài chính dạng markdown cùng metadata (công ty, năm, ngành).

**Các bước**:

1. **Trích xuất bảng**: Parse markdown table thành ma trận ô.
2. **Xây dựng nút ô (Cell Nodes)**: Mỗi ô trở thành một nút `KGNode` với các thuộc tính:
   - `value`: giá trị số đã được chuẩn hóa đơn vị (tỷ, triệu, nghìn đồng, ...).
   - `header`: tiêu đề hàng/cột đã chuẩn hóa.
   - `row_idx`, `col_idx`.
3. **Chuẩn hóa tiêu đề**: Áp dụng ánh xạ đồng nghĩa (synonym mapping) – hỗ trợ cả tiếng Anh lẫn tiếng Việt – để đưa các nhãn về dạng canonical. Ví dụ: `"Tổng tài sản"`, `"Total Assets"` → canonical `"Total Assets"`.
4. **Khớp mẫu (Template Matching)**: So khớp tập tiêu đề đã chuẩn hóa với **Thư viện Mẫu Kế toán** (xây dựng từ IFRS, GAAP, VAS). Mỗi mẫu định nghĩa:
   - Tập tiêu đề cần có để nhận diện mẫu.
   - Một hoặc nhiều **hàm ràng buộc** $f: \mathbb{R}^n \to \mathbb{R}_{\ge 0}$, mỗi hàm đại diện cho một đồng nhất thức kế toán.
   - Các tham chiếu canonical cho biến (variables), đơn vị mặc định và ngưỡng cảnh báo $\epsilon$.
5. **Tạo nút ràng buộc (Constraint Nodes)**: Khi khớp mẫu thành công, đồ thị được bổ sung các nút ràng buộc $c$. Mỗi nút $c$ kết nối (qua cạnh vô hướng) đến toàn bộ các nút ô tham gia vào phương trình. Nút này lưu trữ:
   - **Hàm đánh giá** `evaluator`: tính residual $f_c(\mathbf{v})$, trong đó $\mathbf{v}$ là vector giá trị được gán từ các ô liên quan.
   - **Trọng số** $\omega_i \in \{+1, -1\}$ cho từng ô, xác định chiều đóng góp.
   - **Ngưỡng** $\epsilon$.
6. **Đồ thị $G_D$**: Đầu ra là một đồ thị không đồng nhất (heterogeneous graph) bao gồm:
   - Nút ô (cell nodes).
   - Nút ràng buộc (constraint nodes) với các cạnh nối tới các ô liên quan.
   - Cạnh vị trí (positional edges) giữa các ô cùng hàng/cột, được dùng làm dự phòng khi không khớp mẫu.

**Ví dụ**: Với mẫu "Balance Sheet", hàm ràng buộc:
$$f_{\text{Assets}}(\mathbf{v}) = v_{\text{Total Assets}} - (v_{\text{Total Liabilities}} + v_{\text{Total Equity}})$$

**Điểm khác biệt chính**: Ràng buộc được biểu diễn ở cấp phương trình toàn cục (hyperedge), không còn là các cạnh đơn lẻ rời rạc.

### 1.2 Mã hóa Đồ thị và Điểm nhất quán Phụ thuộc Truy vấn

**Mã hóa đồ thị** sử dụng Mạng Chú ý Đồ thị (GAT) 2 lớp, có nhận thức về loại cạnh:

1. **Embedding nút ô**: Kết hợp embedding của tiêu đề (từ text encoder như BGE‑M3), giá trị số đã chuẩn hóa, và mã hóa vị trí hàng/cột.
2. **Lan truyền thông điệp**: GATv2 với attention riêng cho từng loại cạnh (constraint edge, positional edge).
3. **Pooling**: Mean-pooling toàn bộ nút ô và nút ràng buộc → vector $\mathbf{h}_G \in \mathbb{R}^d$.

**Điểm nhất quán đồ thị có điều kiện truy vấn**: Thay vì trung bình toàn bộ ràng buộc, ta tính trọng số liên quan đến câu hỏi $Q$ cho từng nút ràng buộc $c$:
$$w_c(Q) = \text{softmax}_{c \in \mathcal{C}} \left( \text{cosine\_sim}\big( \mathbf{q}, \mathbf{d}_c \big) \right)$$
trong đó $\mathbf{q}$ là embedding của câu hỏi, $\mathbf{d}_c$ là embedding của mô tả ràng buộc (ví dụ: "Tổng tài sản = Tổng nợ + Vốn chủ sở hữu"). Điểm nhất quán phụ thuộc truy vấn được định nghĩa:
$$s_{\text{constraint}}(Q, G_D) = \frac{\sum_{c \in \mathcal{C}} w_c(Q) \cdot \exp\left(-\frac{|f_c(\mathbf{v})|}{\max(|v_{\text{ref}}|, \epsilon)}\right)}{\sum_{c \in \mathcal{C}} w_c(Q)}$$
với $v_{\text{ref}}$ là giá trị tham chiếu (thường là vế phải của phương trình). Như vậy, chỉ những ràng buộc thực sự liên quan đến câu hỏi mới ảnh hưởng mạnh đến điểm số, tránh bị coi là “quality prior” đơn thuần.

### 1.3 Truy hồi và Xếp hạng lại

1. **Truy hồi thô**: Dùng text embedding của tài liệu (từ multilingual-e5, BGE‑M3) truy vấn chỉ mục FAISS, lấy $4 \times k$ ứng viên.
2. **Tính điểm kết hợp** cho mỗi cặp $(Q, D)$:
   $$s(Q, D) = \alpha \cdot s_{\text{text}}(Q, D) + \beta \cdot s_{\text{entity}}(Q, D) + \gamma \cdot s_{\text{constraint}}(Q, G_D)$$
   - $s_{\text{text}}$: cosine similarity giữa embedding của $Q$ và văn bản tài liệu.
   - $s_{\text{entity}}$: tỉ lệ khớp chính xác công ty, năm, ngành.
   - $s_{\text{constraint}}(Q, G_D)$: định nghĩa như trên.
   - $\alpha, \beta, \gamma$ được học qua softplus.
3. **Xếp hạng lại** và trả về top-$k$ tài liệu, kèm đồ thị $G_D$ và tập con các ràng buộc được chọn (có $w_c(Q) > 0$).

### 1.4 Huấn luyện với CACL (Constraint-Aware Contrastive Learning)

**Sinh nhiễu cứng CHAP** dựa trên chính các ràng buộc phương trình:

- **CHAP‑A (Arithmetic)**: Sửa giá trị một ô trong đồ thị sao cho $f_c(\mathbf{v}) \neq 0$.
- **CHAP‑S (Scale)**: Nhân/chia giá trị một cột với 1000 để phá vỡ quan hệ số học.
- **CHAP‑E (Entity)**: Thay đổi metadata (công ty, năm) để tạo mẫu âm về thực thể.

**Hàm mất mát**:
$$\mathcal{L}_{\text{CACL}} = \underbrace{\max(0, m - s(Q, D^+) + s(Q, D^-))}_{\text{triplet loss}} + \lambda \cdot \underbrace{(-\log \sigma(-s(Q, D^-))) \cdot \mathbb{1}[D^- \text{ violates constraint}]}_{\text{constraint penalty}}$$

**Các giai đoạn huấn luyện**:
1. **Identity pretraining**: Học phân biệt công ty/năm/ngành.
2. **Structural pretraining**: Tối ưu tham số GAT, calibrate điểm nhất quán.
3. **Joint finetuning**: End‑to‑end với CHAP negatives.

---

## Pha 2: Constraint-Aware Fine-Tuning (CAFT)

Pha này mở rộng nguyên lý ràng buộc sang huấn luyện và suy luận của mô hình sinh. Toàn bộ tín hiệu ràng buộc được tính toán ngoài đồ thị tính toán để đảm bảo khả vi.

### 2.1 Grounding Module (Mô-đun Neo Đậu Bằng chứng)

Trước khi sinh câu trả lời, một mô-đun neo đậu tường minh thực hiện các bước:

1. **Evidence Selector**: Từ tài liệu được truy hồi (top‑1), trích xuất danh sách tất cả các ô và đoạn văn bản dưới dạng các cặp `(label, value, coordinate)`.
2. **Cell Linker**: LLM được yêu cầu (qua prompt) liệt kê các bằng chứng sẽ sử dụng, theo định dạng:
   `EVIDENCE: <label> = <value> (row: X, col: Y)`
   Một parser nhỏ kiểm tra sự tồn tại thực tế của label và tọa độ trong bảng. Nếu không khớp, phản hồi được gửi lại để LLM chọn lại.
3. **Variable Canonicalizer**: Mỗi nhãn (ví dụ: `"Lợi nhuận gộp"`, `"Gross Profit"`) được ánh xạ về biến canonical trong Constraint Library thông qua chính ánh xạ đồng nghĩa đã dùng trong GSR. Điều này đảm bảo khi LLM viết `ANSWER: gross_profit=90`, verifier sẽ hiểu đó là biến `Gross Profit`.
4. **Constraint Selector**: Dựa trên tập biến canonical được liệt kê trong `EVIDENCE`, chọn ra tập con các ràng buộc $C(Q, D)$ từ đồ thị $G_D$ có liên quan. Việc này giới hạn phạm vi kiểm chứng và tăng hiệu năng.

### 2.2 Tạo Dữ liệu Ưu tiên với Phân hạng (Tiered Preference Construction)

**Prompt có cấu trúc** được sử dụng cho cả rollout và suy luận:

```
[INST] Dưới đây là tài liệu tài chính liên quan:
{Table markdown}
{Text narrative}

Ràng buộc kế toán đã kiểm tra:
- f_1: Tổng tài sản = Tổng nợ + Vốn chủ sở hữu (residual=0.001)
- f_2: Lợi nhuận gộp = Doanh thu - Giá vốn (residual=0.000)

Câu hỏi: {question}
Hãy chọn bằng chứng (EVIDENCE: ...), suy luận từng bước, và kết thúc bằng:
ANSWER: <tên_biến>=<giá_trị>; ... [/INST]
```

**Sinh tập ứng viên (Rollout)**: Dùng mô hình cơ sở (chưa fine‑tune) sinh $N$ câu trả lời với nhiệt độ cao. Parse dòng `ANSWER:` để thu được dictionary các biến $V_{\text{pred}}$.

**Tính hàm thưởng kép**:
- $\mathcal{R}_{\text{acc}} \in \{0, 1\}$: 1 nếu đáp án cuối cùng khớp ground truth trong ngưỡng 1%, 0 nếu sai.
- $\mathcal{R}_{\text{cstr}}$: Với tập ràng buộc $C(Q,D)$ đã chọn, tính:
  $$\mathcal{R}_{\text{cstr}} = \frac{1}{|C(Q,D)|} \sum_{c \in C(Q,D)} \exp\left(-\frac{|f_c(V_{\text{pred}})|}{\max(|v_{\text{ref}}|, \epsilon)}\right)$$

**Phân hạng ưu tiên (Tiered Preference)**:

Để đảm bảo không bao giờ hy sinh độ chính xác số học cho sự nhất quán nội tại, chúng tôi áp dụng hệ thống phân hạng 4 tầng:

| Tầng | Điều kiện | Mô tả |
|------|-----------|-------|
| **Tier 1** | $\mathcal{R}_{\text{acc}} = 1 \land \mathcal{R}_{\text{cstr}} \ge 1 - \delta$ | Đúng đáp án, nhất quán kế toán |
| **Tier 2** | $\mathcal{R}_{\text{acc}} = 1 \land \mathcal{R}_{\text{cstr}} < 1 - \delta$ | Đúng đáp án, vi phạm nhẹ ràng buộc |
| **Tier 3** | $\mathcal{R}_{\text{acc}} = 0 \land \mathcal{R}_{\text{cstr}} \ge 1 - \delta$ | Sai đáp án, nhưng suy luận nhất quán |
| **Tier 4** | Còn lại | Sai và vi phạm |

Khi tạo cặp cho DPO, mẫu ở tầng cao hơn luôn được chọn làm $y_{\text{win}}$ so với mẫu ở tầng thấp hơn. Chỉ khi cùng tầng mới so sánh điểm thưởng tổng. Cách làm này đảm bảo **accuracy được ưu tiên tuyệt đối**, trong khi constraint vẫn đóng vai trò phân biệt các suy luận đúng và thúc đẩy tính nhất quán nội tại.

### 2.3 Tinh chỉnh với DPO

Mô hình LLM được tinh chỉnh để tối thiểu hóa mất mát DPO trên các cặp đã xây dựng:
$$\mathcal{L}_{\text{CAFT}} = \mathcal{L}_{\text{DPO}}(\pi_\theta; y_{\text{win}}, y_{\text{lose}})$$
Quá trình này hoàn toàn khả vi vì $\mathcal{R}_{\text{total}}$ được tính offline và chỉ dùng để gán nhãn cặp ưu tiên.

### 2.4 Suy luận với Tự Sửa lỗi (Inference‑Time Self‑Correction)

Khi suy luận, pipeline thực hiện tuần tự:

1. **Grounding**: Mô-đun neo đậu (2.1) xác định bằng chứng, chuẩn hóa biến và chọn tập ràng buộc $C(Q,D)$.
2. **Generation**: LLM sinh câu trả lời có cấu trúc `ANSWER:`.
3. **Verification**: Constraint Verifier thực hiện ba kiểm tra:
   - **Evidence consistency**: Đối chiếu `EVIDENCE` của LLM với dữ liệu thực tế trong tài liệu (label có tồn tại? giá trị trích xuất có khớp?).
   - **Variable mapping**: Đảm bảo mọi biến trong `ANSWER:` được ánh xạ thành công về canonical trong Constraint Library.
   - **Equation satisfaction**: Tính $f_c(V_{\text{pred}})$ cho mọi $c \in C(Q,D)$. Nếu $\max_c |f_c| > \tau$, kích hoạt **Self‑Correction**: một prompt phản hồi chứa thông tin lỗi (ràng buộc nào bị vi phạm, giá trị residual) được gửi lại cho LLM để sinh lại. Quá trình lặp tối đa $K$ vòng.

Constraint Verifier sử dụng chính các hàm đánh giá $f_c$ từ Thư viện Ràng buộc Kế toán của Pha 1, đảm bảo tính nhất quán tuyệt đối xuyên suốt pipeline.

---

## Tóm tắt Các Thành phần Chính và Vai trò

| Thành phần | Mô tả | Vai trò trong Pipeline |
|-----------|-------|------------------------|
| **Thư viện Ràng buộc Kế toán** | Tập phương trình biểu tượng (có biến canonical, đơn vị, hàm đánh giá) | Cung cấp tri thức miền dùng chung cho cả hai pha |
| **Nút ràng buộc (Constraint Nodes)** | Biểu diễn phương trình trong đồ thị tài liệu | Cho phép tính điểm nhất quán cấp phương trình ở Pha 1, tạo cơ sở cho reward và verifier ở Pha 2 |
| **$s_{\text{constraint}}(Q, G_D)$** | Điểm nhất quán có trọng số theo truy vấn | Tín hiệu xếp hạng lại trong GSR; cơ sở cho $\mathcal{R}_{\text{cstr}}$ |
| **CHAP Sampler** | Sinh nhiễu cứng dựa trên ràng buộc | Huấn luyện CACL phân biệt được các lỗi cấu trúc tài chính |
| **Tiered Preference (Phân hạng ưu tiên)** | Hệ thống 4 tầng ưu tiên accuracy trước, constraint sau | Xây dựng cặp DPO an toàn, không đánh đổi độ chính xác |
| **Grounding Module** | Neo đậu bằng chứng, chuẩn hóa biến, chọn ràng buộc | Đảm bảo LLM dùng đúng dữ liệu và verifier làm việc chính xác |
| **Constraint Verifier** | Tái sử dụng các hàm $f_c$ để kiểm tra evidence, biến, phương trình | Tự động phát hiện vi phạm và kích hoạt tự sửa lỗi |