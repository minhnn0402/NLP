# Đề tài nghiên cứu: Kiến trúc LLM Mixin KG với tri thức miền cho bài toán End-to-End RAG trên domain tài chính

## Paper story

Paper trình bày một kiến trúc RAG mới cho domain tài chính, trong đó khả năng reasoning của LLM được kết hợp với tri thức miền được mã hoá bằng Knowledge Graph (KG). Ý tưởng chính là: retrieval không chỉ dựa vào semantic similarity, mà còn phải hiểu các ràng buộc tài chính như tổng tài sản, doanh thu, lợi nhuận, dòng tiền, tỷ lệ tài chính; generation không chỉ sinh câu trả lời từ context, mà phải reasoning trên evidence đã retrieve và các constraint tài chính đi kèm.

Đóng góp thứ hai là xây dựng bộ dữ liệu financial domain cho tiếng Việt, phục vụ bài toán End-to-End RAG trên tài liệu tài chính tiếng Việt. Bộ dữ liệu này giúp kiểm tra liệu các kiến trúc RAG hiện tại có thật sự hoạt động tốt trong bối cảnh có bảng số liệu, metadata doanh nghiệp, luật kế toán và câu hỏi cần suy luận số học hay không.

Thông điệp chính của paper: phương pháp đề xuất vượt trội trên cả tiếng Việt lẫn tiếng Anh. Trên tiếng Việt, paper chứng minh mô hình khai thác tốt tri thức miền và bộ law/rule tài chính bản địa hoá. Trên tiếng Anh, paper chạy lại trên các benchmark chuẩn để chứng minh phương pháp không chỉ overfit vào dataset tiếng Việt mà còn cạnh tranh với SOTA retrieval/domain methods.

## Các đóng góp chính

1. Xây dựng bộ dữ liệu financial-domain RAG cho tiếng Việt, gồm câu hỏi, context tài chính dạng text + table, metadata doanh nghiệp và ground-truth evidence/document.

2. Đề xuất kiến trúc End-to-End RAG có 2 phase chính:

### Phase 1: Retrieve Only

Phase này đã có source code trong `ours/source`.

Đóng góp của Phase 1 là một retriever có tri thức miền:

- Mã hoá các luật/identity tài chính thành KG.
- Dùng template/law tài chính kiểu IFRS/GAAP/VAS để nhận diện quan hệ giữa các cột/bảng.
- Dựng constraint KG từ bảng tài chính.
- Kết hợp KG với retrieve model như BGE-M3, multilingual-e5, BGE-large hoặc các retriever mạnh khác.
- Sinh hard negative bằng cách phá vỡ constraint tài chính.
- Huấn luyện retriever bằng contrastive learning có awareness về constraint.

Mục tiêu: retrieve đúng tài liệu/bảng/evidence ngay cả khi lexical overlap thấp, hoặc khi có hard negative rất giống về text nhưng sai entity, sai năm, sai scale hoặc sai công thức kế toán.

### Phase 2: Generate / LLM Reasoning

Phase này chưa có code, hiện là idea cần phát triển trong proposal.

Đóng góp mong muốn là một kiến trúc LLM reasoning nhận các candidate từ Phase 1, sau đó sinh câu trả lời có căn cứ bằng cách kết hợp:

- Retrieved evidence text/table.
- KG/constraint graph của từng candidate.
- Metadata như công ty, năm báo cáo, ngành.
- Các luật tài chính liên quan.
- Chain-of-thought hoặc structured reasoning được kiểm soát.

Mục tiêu: LLM không chỉ đọc context rồi trả lời, mà phải biết kiểm tra consistency số học, chọn đúng entity/year, dùng đúng luật tài chính và giải thích được nguồn evidence.

## Chi tiết từng thành phần

### 1. Dataset cho tiếng Việt

Phần này đã làm nhưng chưa đưa vào source hiện tại.

Dataset tiếng Việt nên được trình bày như một đóng góp độc lập:

- Nguồn dữ liệu: báo cáo tài chính, báo cáo thường niên, tài liệu công bố của doanh nghiệp, hoặc dữ liệu tài chính đã chuẩn hoá.
- Dạng context: narrative text + bảng tài chính.
- Dạng question: câu hỏi về chỉ số, so sánh, thay đổi theo năm, truy xuất entity, truy xuất bảng, hoặc suy luận từ nhiều ô.
- Metadata: tên công ty, năm báo cáo, ngành, loại báo cáo, loại bảng.
- Ground truth: document/evidence/table/cell hoặc answer tuỳ theo task.

Các task có thể chia thành:

- Retrieve-only: câu hỏi cần tìm đúng document/evidence.
- QA/generation: câu hỏi cần sinh câu trả lời cuối.
- Numerical reasoning: câu hỏi cần tính toán hoặc kiểm tra quan hệ số học.
- Entity/year-sensitive QA: câu hỏi dễ nhầm giữa công ty/năm tương tự.

Điểm cần nhấn mạnh: dataset tiếng Việt không chỉ là bản dịch, mà cần phản ánh đặc thù tài liệu tài chính tiếng Việt, cách gọi chỉ tiêu kế toán tiếng Việt và chuẩn VAS/IFRS dùng trong báo cáo.

### 2. Retrieve Only từ source code

Source hiện tại hiện thực Phase 1 với tên GSR-CACL: Graph-Structured Retrieval + Constraint-Aware Contrastive Learning.

#### 2.1 GSR: Graph-Structured Retrieval

Pipeline chính:

1. Mỗi document gồm text, markdown table và metadata.
2. Trích bảng từ document.
3. Parse bảng thành các cell node.
4. Normalize header và match với thư viện 15 template tài chính/kế toán.
5. Dựng constraint KG:
   - Node: từng ô bảng, có value, text, header, row/column.
   - Edge: accounting edge hoặc positional edge.
   - `omega = +1`: quan hệ cộng.
   - `omega = -1`: quan hệ trừ.
   - `omega = 0`: quan hệ vị trí fallback.
6. Dùng GAT encoder để mã hoá KG.
7. FAISS retrieve candidate theo text embedding.
8. Joint scorer rerank candidate bằng text similarity, entity score và constraint score.

Joint score:

```text
s(Q, D) = alpha * s_text(Q, D)
        + beta  * s_entity(Q, D)
        + gamma * s_constraint(G_D)
```

Trong đó:

- `s_text`: similarity giữa query và document.
- `s_entity`: match company/year/sector.
- `s_constraint`: độ nhất quán của bảng với financial KG.

#### 2.2 KG và luật tài chính

Source hiện dùng 15 template IFRS/GAAP phổ biến:

- income statement
- balance sheet assets
- balance sheet liabilities/equity
- cash flow
- revenue segment
- quarterly breakdown
- EPS
- EBITDA
- debt schedule
- shareholder equity
- gross margin
- operating margin
- net margin
- current ratio

Khi mở rộng sang tiếng Việt, phần này cần được chuyển thành bộ financial law/rule tiếng Việt:

- Dịch hoặc dựng lại template theo VAS/IFRS.
- Bổ sung synonym mapping tiếng Việt cho tên chỉ tiêu.
- Chuẩn hoá đơn vị: đồng, nghìn đồng, triệu đồng, tỷ đồng.
- Chuẩn hoá cách viết số âm, phần trăm, dấu phẩy/dấu chấm.

#### 2.3 CACL và hard negative

Source có CHAP negative sampler để tạo hard negative theo tri thức tài chính:

- `CHAP-A`: sửa một ô để phá quan hệ cộng/trừ.
- `CHAP-S`: đổi scale như triệu/tỷ để phá quan hệ số học.
- `CHAP-E`: đổi company/year để tạo negative sai entity.

Loss training:

```text
L_CACL = L_triplet + lambda * L_constraint
```

Ý nghĩa: positive phải được score cao hơn hard negative, đồng thời negative vi phạm constraint phải bị phạt nếu được score cao.

#### 2.4 Training retrieve model

Source hỗ trợ 3 stage:

1. Identity pretraining: học phân biệt company/year/sector.
2. Structural pretraining: học constraint/KG path.
3. Joint finetuning: huấn luyện end-to-end với CHAP negatives.

Retriever có thể kết hợp với BGE-large, multilingual-e5, BGE-M3 hoặc các embedding/retrieval model khác. Với tiếng Việt, nên ưu tiên multilingual retriever mạnh và fine-tune trên dataset financial-domain tiếng Việt.

### 3. LLM Reasoning

Phần này chưa nghĩ kỹ và chưa có source code. Có thể phát triển theo hướng sau.

#### 3.1 Input cho LLM

LLM nhận:

- Query gốc.
- Top-k candidate từ GSR.
- Evidence text/table đã retrieve.
- Metadata candidate.
- Constraint KG hoặc tóm tắt constraint liên quan.
- Các law/template được match.

Thay vì đưa context thô, có thể đưa một structured evidence package:

```text
Question
Retrieved evidence
Matched financial template
Relevant rows/cells
Constraint checks
Entity/year metadata
```

#### 3.2 Reasoning method

Có thể thiết kế LLM reasoning theo 2 tầng:

1. Evidence selection: LLM chọn candidate/table/cell nào thật sự liên quan.
2. Answer generation: LLM sinh câu trả lời dựa trên evidence đã chọn và constraint check.

Các dạng reasoning cần hỗ trợ:

- Entity grounding: câu hỏi hỏi công ty/năm nào thì chỉ dùng evidence đúng entity/năm đó.
- Table grounding: xác định hàng/cột liên quan.
- Numerical reasoning: tính toán hoặc so sánh từ các ô.
- Constraint verification: kiểm tra đáp án có vi phạm law/identity không.
- Citation: trả lời kèm evidence/cell/table/document.

#### 3.3 Training LLM/generator

Một số hướng training có thể viết trong proposal:

- Supervised fine-tuning trên bộ QA tiếng Việt có answer + evidence.
- Instruction tuning với format đầu vào có KG/evidence package.
- Preference optimization: answer đúng evidence và đúng constraint được prefer hơn answer hallucinate.
- Distillation từ LLM mạnh sang model nhỏ hơn cho financial QA.
- Multi-task learning: answer generation + evidence selection + constraint verification.

#### 3.4 Điểm mới của Phase 2

LLM Mixin KG không chỉ là "retrieve rồi nhét context vào LLM", mà là:

- KG-guided evidence packaging.
- Constraint-aware decoding hoặc verification.
- LLM reasoning có grounding vào law/template tài chính.
- Có thể reject hoặc flag answer khi evidence không đủ hoặc constraint check fail.

## Các công việc phải làm

### Main result

#### ii) Tiếng Việt

Xây dựng bộ law/rule tài chính cho tiếng Việt:

- Dịch hoặc dựng lại bộ law kiểu IFRS/VAS.
- Xây dựng synonym mapping cho chỉ tiêu tài chính tiếng Việt.
- Chuẩn hoá đơn vị và format số trong báo cáo tiếng Việt.
- Tạo template KG cho các loại bảng phổ biến.

Chạy lại các SOTA method tiếng Anh trên dataset tiếng Việt:

- BM25 / HybridBM25.
- Dense retriever multilingual.
- BGE-M3 / multilingual-e5 / các retriever general mạnh.
- Domain retriever nếu có.
- Ours: GSR-CACL + law/KG tiếng Việt.

Mục đích kể chuyện: các method general không đủ hiểu domain tài chính tiếng Việt; ours best vì đưa tri thức miền vào cả retrieve và training.

#### iii) Tiếng Anh

Chạy method trên dữ liệu tiếng Anh:

- FinQA.
- ConvFinQA.
- TAT-DQA.
- T2-RAGBench hoặc benchmark retrieve-only phù hợp.

Chạy kèm:

- SOTA general retriever.
- SOTA domain retriever.
- BM25/HybridBM25.
- Dense retriever.
- ColBERT-style retriever nếu có.

Mục đích kể chuyện: method không chỉ tốt trên tiếng Việt, mà còn vượt hoặc cạnh tranh mạnh với các SOTA hiện tại trên benchmark tiếng Anh.

### Appendix

#### 1. Chứng minh Phase 1 SOTA trên retrieve-only

Phần này có thể dùng source hiện tại làm bằng chứng chính.

Các benchmark:

- T2-RAGBench / FinQA.
- T2-RAGBench / ConvFinQA.
- T2-RAGBench / TAT-DQA.

Metrics:

- MRR@3.
- Recall@1/3/5.
- NDCG@3.

Source hiện đã có một run FinQA GSR:

```text
MRR@3    ≈ 0.604
Recall@3 ≈ 0.650
NDCG@3   ≈ 0.616
```

#### 2. Ablation study

Ablation cần có:

- Ours full.
- w/o KG.
- w/o constraint score.
- w/o entity score.
- w/o CHAP, thay bằng random negatives.
- w/o structural pretraining.
- w/o identity pretraining.
- GSR vs HybridGSR.
- Different retrievers: BGE-M3, multilingual-e5, BGE-large.
- Different law/template coverage.

#### 3. Dataset/law analysis

Nên có thêm appendix:

- Coverage của financial template trên dataset tiếng Anh.
- Coverage của law/template tiếng Việt.
- Error analysis các case sai do entity/year.
- Error analysis các case sai do numerical/scale.
- Case study LLM reasoning có và không có KG/constraint.

## Tóm tắt positioning

Source code hiện tại là phần hiện thực mạnh cho Phase 1 Retrieve Only. Proposal tổng thể lớn hơn source: nó đặt GSR-CACL vào một kiến trúc End-to-End RAG cho tài chính, thêm dataset tiếng Việt và Phase 2 LLM Reasoning. Khi viết paper, không nên trình bày source như toàn bộ đề tài, mà nên dùng nó như module retriever trong kiến trúc LLM Mixin KG.
