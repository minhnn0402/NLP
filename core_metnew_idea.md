# LEDGER: Truy hồi và Suy luận ở cấp Dữ kiện cho Hỏi–đáp Tài chính

## Tóm tắt

Hỏi–đáp trên báo cáo tài chính đòi hỏi tìm đúng đoạn bằng chứng trong những tài liệu vừa có văn bản vừa có bảng số, rồi suy luận số chính xác trên bằng chứng đó; câu trả lời hầu như luôn là một con số gắn với một *doanh nghiệp*, một *kỳ báo cáo* và một *chỉ tiêu kế toán* xác định. Trên bộ dữ liệu chuẩn T²-RAGBench, các bộ truy hồi dựa trên vector ngữ nghĩa (dense retrieval) thua cả BM25, và khoảng cách giữa hiệu năng khi đã có sẵn đúng ngữ cảnh (khoảng 72% MRR@3) với hiệu năng khi phải tự truy hồi (khoảng 40%) cho thấy điểm nghẽn nằm ở cả hai khâu *truy hồi* và *sinh*.

Chúng tôi cho rằng nguyên nhân sâu xa là: thông tin quyết định một tài liệu có liên quan hay không, cũng như tính đúng đắn của một bước suy luận, đều nằm ở những **chi tiết rất cụ thể** — đúng doanh nghiệp, đúng năm, đúng con số — nhưng các mô hình nhúng (embedding) lại làm mờ những chi tiết này, còn các mô hình sinh thì dễ bịa con số hoặc tính sai. Hai khó khăn đã được chỉ ra độc lập trong các nghiên cứu gần đây: (i) một vector tóm tắt duy nhất không thể vừa nắm bắt chủ đề tổng thể vừa giữ được các chi tiết phân biệt nhỏ; và (ii) các con số bị tách vụn khi mã hóa, khiến mô hình gần như không phân biệt được giá trị của chúng.

Chúng tôi đề xuất **LEDGER**, một khung hỏi–đáp tài chính xây dựng trên một nhận định xuyên suốt: *độ liên quan và tính đúng đắn cần được xét ở cấp từng dữ kiện cụ thể, chứ không phải ở cấp toàn bộ tài liệu.* Trung tâm của khung là một **tập dữ kiện (Fact Ledger)**: mỗi tài liệu được biểu diễn thành một tập **dữ kiện (fact)** có cấu trúc gồm khái niệm, doanh nghiệp, kỳ và giá trị. Cùng một tập dữ kiện này phục vụ ba đóng góp gắn bó chặt chẽ:

1. **Truy hồi ở cấp dữ kiện** với biểu diễn tách riêng ba luồng (ngữ nghĩa khái niệm, độ lớn con số, khóa định danh) và cơ chế *đối sánh muộn có cổng*, giúp những chi tiết phân biệt không bị nhấn chìm.
2. **Sinh mẫu âm khó từ chính tập dữ kiện**, khớp theo từng luồng và chắc chắn làm câu trả lời không còn đúng, qua đó huấn luyện bộ truy hồi mà không gặp nhiễu "mẫu âm giả".
3. **Huấn luyện mô hình sinh bằng phần thưởng quá trình do tập dữ kiện kiểm chứng**: chính tập dữ kiện đóng vai trò một bộ kiểm chứng mang tính ký hiệu, tất định, không cần gán nhãn, chấm điểm từng bước suy luận theo mức độ *bám nguồn giá trị* và *nhất quán số học*, qua đó dạy mô hình suy luận có căn cứ thay vì đoán đúng đáp án một cách may rủi.

Ba đóng góp chia sẻ cùng một biểu diễn và cùng một nguyên lý: *lập chỉ mục theo dữ kiện*. Chúng tôi trình bày kế hoạch đánh giá trên T²-RAGBench (FinQA, ConvFinQA, TAT-DQA) cho cả hai khâu truy hồi và sinh, cùng các phân tích nhằm chỉ rõ nguồn gốc của phần hiệu năng cải thiện.

---

## 1. Giới thiệu

### 1.1. Bối cảnh và động lực

Tài liệu tài chính — báo cáo thường niên, hồ sơ gửi cơ quan quản lý, báo cáo kết quả kinh doanh — có đặc thù riêng: thông tin quan trọng nằm trong các *bảng số* xen kẽ *văn bản diễn giải*, và gần như mọi câu hỏi đều cần một con số được xác định rõ theo doanh nghiệp, năm tài chính và chỉ tiêu kế toán. Hỏi–đáp ở đây gồm hai khâu nối tiếp: *truy hồi* tài liệu chứa bằng chứng, rồi *sinh* lời giải số nhiều bước (trích vài con số rồi tính hiệu, tỉ lệ, phần trăm thay đổi…). Cả hai khâu đều phải chính xác thì câu trả lời mới đúng.

Bộ dữ liệu T²-RAGBench đánh giá toàn bộ chuỗi này trong điều kiện **không cho sẵn ngữ cảnh**. Ba kết quả từ đó đáng chú ý. *Thứ nhất*, truy hồi là một điểm nghẽn lớn: khi được cung cấp sẵn đúng ngữ cảnh, độ chính xác đạt khoảng 72%, nhưng khi phải tự truy hồi thì chỉ còn khoảng 40% MRR@3. *Thứ hai*, các bộ truy hồi vector ngữ nghĩa mạnh nhất vẫn thua phương pháp lai với BM25 — trái với quan niệm phổ biến. *Thứ ba*, ngay cả khi đã có ngữ cảnh, khâu sinh vẫn mắc lỗi suy luận số, và dư địa cải thiện ở cả hai khâu còn rất lớn.

### 1.2. Vì sao các phương pháp hiện tại thất bại

**Ở khâu truy hồi**, hai hiện tượng đã được ghi nhận trong các công trình gần đây cùng giải thích thất bại.

- *Mâu thuẫn giữa khái quát và chi tiết (granularity dilemma).* Khi nén cả tài liệu vào một vector duy nhất, mô hình buộc phải đánh đổi: hoặc bám sát chủ đề tổng thể, hoặc giữ lại các chi tiết phân biệt nhỏ như doanh nghiệp, năm, con số cụ thể. Trên thực tế các chi tiết này bị "trung bình hóa" và mất đi.
- *Hạn chế xử lý con số (numeracy gap).* Các mô hình nhúng gần như không phân biệt được giá trị của các con số: khi mã hóa, một số như "3.035" bị tách thành nhiều mảnh làm vỡ thông tin về độ lớn; đồng thời phần ngữ nghĩa xung quanh chiếm gần hết "dung lượng" biểu diễn, khiến những con số khác nhau bị ánh xạ về các vector gần như trùng nhau.

Hậu quả đo được: tài liệu *sai* thường được chấm điểm cao hơn tài liệu *đúng*; các báo cáo của *cùng một doanh nghiệp* tương đồng gấp nhiều lần so với báo cáo khác doanh nghiệp, khiến việc phân biệt cùng doanh nghiệp nhưng khác năm đặc biệt khó; và vì mỗi tài liệu chứa hàng chục con số, cùng một giá trị xuất hiện ở rất nhiều tài liệu nên con số vừa là manh mối vừa là cái bẫy. Thêm vào đó, hầu hết tài liệu tài chính dùng chung một vốn thuật ngữ rất lớn, nên gần như không thể phân biệt tài liệu chỉ dựa vào mặt chữ.

**Ở khâu sinh**, suy luận số nhiều bước cũng dễ sai theo những cách mà các cách huấn luyện hiện tại chưa xử lý tốt. Học tăng cường chỉ dựa trên *đáp án cuối* cho tín hiệu thưa và dễ bị "lách": mô hình có thể ra đúng số nhờ may rủi hoặc nhờ các bước sai tự triệt tiêu nhau — tức "đúng đáp án nhưng sai lập luận". Ngược lại, bộ thưởng theo từng bước (process reward model) cho tín hiệu dày nhưng cần *gán nhãn từng bước* rất tốn kém, và bản thân bộ thưởng học được cũng có thể bị đánh lừa. Các phương pháp thưởng "tính trung thực" gần đây chủ yếu kiểm tra *văn bản* có bám nguồn hay không, bằng một bộ thưởng học được, và không kiểm tra *giá trị số* hay *tính đúng của phép tính*.

### 1.3. Nhận định cốt lõi

Điểm chung của mọi hiện tượng trên là: **độ liên quan của tài liệu, cũng như tính đúng đắn của một bước suy luận, đều được xác định ở cấp từng dữ kiện, không phải ở cấp toàn tài liệu.** Tài liệu liên quan đến một câu hỏi là tài liệu chứa đúng *dữ kiện* — một bộ (khái niệm, doanh nghiệp, kỳ, giá trị) — trả lời cho câu hỏi; một bước suy luận đáng tin là bước đặt nền trên các dữ kiện có thật và thực hiện phép tính nhất quán. Vì vậy, lời giải là **mô hình hóa các thành phần của dữ kiện một cách tường minh và độc lập, rồi dùng chính chúng làm chuẩn ở cả khâu truy hồi lẫn khâu sinh.**

### 1.4. Đóng góp

Chúng tôi hiện thực hóa nhận định trên trong khung **LEDGER**, với ba đóng góp gắn kết chặt chẽ quanh một biểu diễn chung là tập dữ kiện.

1. **Truy hồi ở cấp dữ kiện với biểu diễn tách riêng và định tuyến theo doanh nghiệp – thời gian (§5.2–5.5).** Chúng tôi tách ba luồng — ngữ nghĩa khái niệm, độ lớn con số, khóa định danh — và so khớp bằng *đối sánh muộn có cổng*: tương đồng khái niệm được nhân với các cổng doanh nghiệp – thời gian, biến tín hiệu phân biệt thành một thừa số có tính quyết định thay vì một số hạng cộng dễ bị lấn át. Để chạy được ở quy mô lớn và không "dễ gãy", chúng tôi gộp bước lọc định danh vào ngay quá trình duyệt chỉ mục vector, kèm phương án dự phòng nới lỏng dần.

2. **Sinh mẫu âm khó từ chính tập dữ kiện, khớp theo từng luồng (§5.6).** Chúng tôi tạo mẫu âm bằng các phép biến đổi có chủ đích trên *dữ kiện chứa đáp án* của tài liệu đúng. Dưới một giả định được phát biểu rõ ràng, các mẫu âm này **làm câu trả lời không còn đúng**, nên chắc chắn hợp lệ và không gây nhiễu "mẫu âm giả"; đồng thời mỗi phép biến đổi chỉ chạm đến đúng một luồng thông tin, tức đúng một cổng/kênh của bộ chấm điểm ở Đóng góp 1.

3. **Huấn luyện mô hình sinh bằng phần thưởng quá trình do tập dữ kiện kiểm chứng (§5.7).** Chính tập dữ kiện đóng vai trò một bộ kiểm chứng *mang tính ký hiệu, tất định, không cần gán nhãn*, chấm điểm từng bước suy luận theo mức độ bám nguồn giá trị và nhất quán số học. Dùng tín hiệu dày này trong học tăng cường, mô hình được dạy suy luận có căn cứ thay vì đoán đúng đáp án một cách may rủi — lấp đúng khoảng trống mà học theo đáp-án-cuối và bộ thưởng-học-được để lại.

Ba đóng góp không tách rời và cùng chia sẻ một nguyên lý — *lập chỉ mục theo dữ kiện* — áp dụng lần lượt cho độ liên quan (Đóng góp 1), cho việc tạo mẫu âm huấn luyện (Đóng góp 2), và cho việc giám sát từng bước suy luận (Đóng góp 3). Mối gắn kết ba chiều này được làm rõ ở §5.8.

---

## 2. Các công trình liên quan

**Truy hồi bằng vector ngữ nghĩa và giới hạn.** Các bộ truy hồi nén tài liệu thành một vector tối ưu cho tương đồng ngữ nghĩa tổng thể; trên miền tài chính, từ vựng dùng chung lớn và con số chiếm ưu thế khiến chúng dễ bị đánh lừa. Các nghiên cứu gần đây chỉ ra cả mâu thuẫn khái quát–chi tiết lẫn hạn chế xử lý con số là điểm yếu mang tính hệ thống.

**Truy hồi và suy luận trên bảng tài chính.** Các bộ dữ liệu FinQA, TAT-QA, ConvFinQA định hình bài toán suy luận số trên dữ liệu bảng – văn bản; nhiều mô hình theo sau (đồ thị phân cấp theo ngữ nghĩa, Program-of-Thought, công cụ ngoài) đóng vai trò *bộ đọc hiểu* khi đã có ngữ cảnh. Một số công trình gần đây dùng đồ thị tri thức với lược đồ rất gần với tập dữ kiện của chúng tôi (khái niệm, doanh nghiệp, kỳ, giá trị), nhưng chỉ dùng nó để *truy hồi/làm giàu ngữ cảnh trước khi nhắc* — mô hình sinh được giữ đông cứng, không huấn luyện. LEDGER khác ở hai chỗ: dùng biểu diễn dữ kiện cho *đối sánh có cổng* khi truy hồi, và dùng nó làm *tín hiệu thưởng để huấn luyện* mô hình sinh.

**Truy hồi có tính đến đại lượng và biểu diễn con số.** Một số nghiên cứu đưa thông tin đại lượng vào truy hồi nhưng giả định *điều kiện số nằm tường minh trong câu hỏi*; bài toán của chúng tôi có con số nằm trong *tài liệu* và mẫu âm khó là các báo cáo cùng doanh nghiệp khác năm. Các phương pháp mã hóa độ lớn liên tục cho mô hình ngôn ngữ truyền cảm hứng cho luồng độ lớn tách riêng của chúng tôi.

**Đối sánh muộn (late interaction).** Các mô hình đối sánh muộn giữ biểu diễn theo từng đơn vị và chấm điểm bằng tổng các giá trị tương đồng cực đại; LEDGER nâng ý tưởng này lên *cấp dữ kiện có cấu trúc*, thêm cổng doanh nghiệp – thời gian và luồng độ lớn.

**Khai thác mẫu âm khó và mẫu âm giả.** Chọn mẫu âm theo độ tương đồng có nguy cơ cao chọn phải mẫu âm giả, buộc dùng bộ lọc tốn kém; LEDGER tránh bằng cách *tự tạo* mẫu âm qua biến đổi có chủ đích, đảm bảo câu trả lời không còn đúng.

**Huấn luyện suy luận với phần thưởng kiểm chứng được.** Học tăng cường với phần thưởng kiểm chứng được (RLVR), thường tối ưu bằng GRPO, đang là hướng chủ đạo để nâng năng lực suy luận; tuy nhiên phần thưởng theo *đáp án cuối* là thưa và có thể bị "lách". Bộ thưởng theo từng bước (process reward model) cho tín hiệu dày hơn nhưng vướng nút thắt gán nhãn, và các phương án sinh nhãn tự động (mô phỏng Monte Carlo) khó kiểm soát loại lỗi và vị trí lỗi. Các phương pháp thưởng tính trung thực trong RAG chủ yếu kiểm tra *văn bản* bám nguồn bằng một bộ thưởng học được, trên các tập đa bước văn bản, không nhắm tới *kiểm chứng số học*. LEDGER khác biệt ở chỗ rút ra một phần thưởng quá trình *tất định, ký hiệu, miễn gán nhãn, chuyên cho suy luận số*, ngay từ chính tập dữ kiện đã dùng để truy hồi.

---

## 3. Phát biểu bài toán và Biểu diễn nền

### 3.1. Bài toán

Cho kho tài liệu tài chính $\mathcal D=\{d_1,\dots,d_M\}$, mỗi tài liệu gồm văn bản và bảng kèm thông tin định danh (doanh nghiệp, năm, ngành). Cho câu hỏi $q$, hệ thống phải (a) truy hồi tài liệu chứa bằng chứng, và (b) sinh câu trả lời số kèm lập luận. Độ đo truy hồi: MRR@k, Recall@k, nDCG@k; độ đo sinh: độ chính xác con số và tỉ lệ vi phạm ràng buộc.

### 3.2. Biểu diễn nền: tập dữ kiện tài chính (Fact Ledger)

**Định nghĩa 1 (Dữ kiện).** Một *dữ kiện* (fact) là bộ năm thành phần
$$
f=(m,\,e,\,t,\,v,\,u),
$$
trong đó $m$ là khái niệm kế toán đã chuẩn hóa (ví dụ "Vốn chủ sở hữu"), kèm vector ngữ nghĩa $\mathbf c_f\in\mathbb R^{d}$; $e$ là doanh nghiệp; $t$ là kỳ báo cáo; $v\in\mathbb R$ là giá trị; $u$ là đơn vị/thang đo, dùng để quy về độ lớn chuẩn hóa $\mu_f=\operatorname{sign}(v)\cdot\log_{10}(1+|v|/s_u)$.

**Định nghĩa 2 (Tập dữ kiện của tài liệu).** $\mathcal L(d)=\{f_1,\dots,f_{n_d}\}$ trích từ tài liệu $d$: mỗi ô số trong bảng sinh một dữ kiện với khái niệm lấy từ *nhãn dòng*, kỳ lấy từ *nhãn cột*, doanh nghiệp lấy từ định danh hoặc văn bản xung quanh; các con số có ngữ cảnh trong văn bản diễn giải cũng được trích.

**Định nghĩa 3 (Ý định câu hỏi).** Câu hỏi được phân tích thành các "ô cần điền" $\mathcal Q(q)=\{(\tilde m_j,\mathrm{op}_j,\tilde v_j)\}_{j=1}^{k}\cup(\tilde e,\tilde t)$, với $\tilde m_j$ là (các) chỉ tiêu được hỏi, $(\tilde e,\tilde t)$ là doanh nghiệp và kỳ, $(\mathrm{op}_j,\tilde v_j)$ là phép so sánh/giá trị nếu có.

Tập dữ kiện là *nền chung*: được chấm điểm ở Đóng góp 1, bị biến đổi ở Đóng góp 2, và làm bộ kiểm chứng ở Đóng góp 3.

---

## 4. Phân tích: khi nào truy hồi một-vector thất bại

**Mô hình cộng tính (xấp xỉ ban đầu).** Trong một nhóm tài liệu gần như trùng nhau, xấp xỉ vector tài liệu bằng tổng một thành phần *chủ đề* (chia sẻ rộng theo ngành và vốn từ chỉ tiêu) và một thành phần *định danh* phụ thuộc $(e,t)$:
$$
\mathbf h_d\approx \mathbf u_{\text{chủđề}}+\mathbf u_{\text{địnhdanh}}(e,t)+\boldsymbol\varepsilon.
$$

**Quan sát.** Với tài liệu đúng $d^\star=(e,t)$ và báo cáo cùng doanh nghiệp khác năm $d^-=(e,t')$, chênh lệch điểm tích vô hướng là
$$
\Delta=\big\langle \mathbf u_{\text{địnhdanh}}(\tilde e,\tilde t),\ \mathbf u_{\text{địnhdanh}}(e,t)-\mathbf u_{\text{địnhdanh}}(e,t')\big\rangle+\text{nhiễu},
$$
vì hai thành phần chủ đề triệt tiêu nhau. Nếu độ phân tán của thành phần chủ đề lớn hơn hẳn của thành phần định danh, $\Sigma_{\text{chủđề}}\gg\Sigma_{\text{địnhdanh}}$, thì tỉ số tín hiệu trên nhiễu của bộ phân biệt một-vector thỏa $\mathrm{SNR}\lesssim \Sigma_{\text{địnhdanh}}/\Sigma_{\text{chủđề}}\to 0$: khả năng phân biệt đúng/sai theo doanh nghiệp – năm tiến về mức ngẫu nhiên khi vốn từ chủ đề càng dùng chung — đúng đặc thù tài chính.

**Kiểm chứng thực nghiệm.** Chúng tôi (i) huấn luyện bộ dò tuyến tính dự đoán $(e,t)$ và dự đoán chủ đề (ngành) từ vector tài liệu, kỳ vọng độ chính xác cho chủ đề cao vượt trội; và (ii) ước lượng trực tiếp $\widehat\Sigma_{\text{chủđề}},\widehat\Sigma_{\text{địnhdanh}}$ bằng cách chiếu vector lên các không gian con tương ứng. Lập luận này dẫn tới quyết định thiết kế: thay điểm cộng tính bằng dạng *nhân* tách bạch các khía cạnh.

---

## 5. Phương pháp

### 5.1. Trích xuất tập dữ kiện

Quy trình gồm: (1) phân tích bảng thành lưới ô, nhận diện trục nhãn (dòng) và trục kỳ (cột) dựa vào vị trí và kiểu dữ liệu, hỗ trợ cả bảng đặt ngược trục; (2) chuẩn hóa khái niệm về dạng thống nhất bằng từ điển đồng nghĩa và quy đổi viết tắt ↔ dạng đầy đủ, kết hợp căn chỉnh bằng vector ngữ nghĩa; (3) chuẩn hóa đơn vị/thang đo để tính độ lớn $\mu_f$; (4) trích dữ kiện số có ngữ cảnh từ văn bản. Toàn bộ chạy ngoại tuyến và lưu sẵn theo tài liệu. Với bảng nhiều tầng tiêu đề hoặc có ô gộp vượt khả năng phân tích, hệ thống lui về biểu diễn dữ kiện ở cấp dòng văn bản, đảm bảo tài liệu không bị loại khỏi tập ứng viên.

### 5.2. Biểu diễn dữ kiện theo ba luồng tách riêng

- **Luồng khái niệm:** $\mathbf c_f=\mathrm{Enc}_\theta(m_f)$, chỉ mã hóa *chuỗi khái niệm ngắn*, để con số thô không làm nhiễu biểu diễn ngữ nghĩa.
- **Luồng độ lớn:** $\mu_f$ giữ riêng ở dạng vô hướng, bảo toàn thông tin độ lớn; chỉ tham gia chấm điểm khi câu hỏi có thành phần số.
- **Khóa định danh:** $e_f,t_f$ đã chuẩn hóa, là tín hiệu phân biệt được đưa lên thành thành phần chấm điểm độc lập.

### 5.3. Đối sánh muộn có cổng ở cấp dữ kiện

Điểm liên quan giữa câu hỏi và tài liệu là tổng, theo từng ô cần điền, của đóng góp từ dữ kiện khớp nhất:
$$
\boxed{\,S(q,d)=\sum_{j=1}^{k}\ \max_{f\in\mathcal L(d)}\Big[\ \sigma_m(\tilde{\mathbf c}_j,\mathbf c_f)\cdot\gamma_e(\tilde e,e_f)\cdot\gamma_t(\tilde t,t_f)\cdot\rho_j(\mu_f)\ \Big]\,}
$$
với $\sigma_m(\mathbf a,\mathbf b)=\mathrm{ReLU}(\cos(\mathbf a,\mathbf b))^{1/\tau_m}$ là tương đồng khái niệm (tham số nhiệt $\tau_m$); $\gamma_t(\tilde t,t)=\exp(-|\tilde t-t|/\sigma_t)$ là cổng thời gian mềm (độ rộng dung sai năm $\sigma_t$); $\gamma_e$ là cổng doanh nghiệp; $\rho_j(\mu_f)$ là hệ số độ lớn, mặc định $1$ khi câu hỏi không nêu số. Phép nhân các cổng khiến một dữ kiện sai năm hoặc sai doanh nghiệp bị *triệt tiêu điểm* bất kể khái niệm khớp đến đâu, nên tín hiệu phân biệt trở thành thừa số quyết định thay vì số hạng cộng dễ bị lấn át. Biểu diễn nhiều vector ở cấp dữ kiện cho phép một tài liệu phục vụ nhiều câu hỏi khác nhau, và cơ chế lấy cực đại để mỗi ô cần điền "chọn" dữ kiện ủng hộ mạnh nhất.

### 5.4. Tìm kiếm một-giai-đoạn có mặt nạ, kèm dự phòng nới lỏng

Tìm thô bằng chỉ mục vector trên khái niệm rồi mới lọc theo định danh sẽ trả về vô số dữ kiện cùng khái niệm từ mọi doanh nghiệp/năm — hoặc làm rơi tài liệu đúng, hoặc làm chi phí bùng nổ. Chúng tôi gộp hai bước: thông tin định danh trở thành một *mặt nạ* can thiệp ngay vào quá trình duyệt chỉ mục lân cận. Một chỉ mục ngược ánh xạ doanh nghiệp $e\simeq\tilde e$ giao với tập năm $\{\tilde t-1,\tilde t,\tilde t+1\}$ thành tập định danh hợp lệ (giữ năm lân cận để cổng $\gamma_t$ mềm còn tác dụng); trong khi duyệt, điểm ngoài mặt nạ bị bỏ qua trước cả khi tính tương đồng. Chi phí đối sánh muộn chỉ tỉ lệ với số ứng viên sau mặt nạ, nên *độ trễ có giới hạn* thay vì tăng tuyến tính theo kho. Để tránh "dễ gãy" khi phân tích câu hỏi sai, hệ thống đếm số phần tử trong mặt nạ: nếu quá ít thì tự nới (bỏ ràng buộc doanh nghiệp, giữ ràng buộc năm); nếu vẫn rỗng thì gỡ mặt nạ và lui về tìm theo khái niệm thuần, để các cổng tự điều chỉnh ở bước sau. Lỗi phân tích nhờ vậy chỉ làm hiệu năng giảm dần.

### 5.5. Truy hồi có tính đến suy luận nhiều bước

Một phần câu hỏi cần nhiều bước (ví dụ phần trăm thay đổi giữa hai năm liền kề), tức cần *hai* dữ kiện cùng chỉ tiêu ở hai kỳ. Bộ phân tích ý định gán phép toán và cấu trúc kỳ; điểm cho ô cần điền khi đó đòi hỏi tài liệu chứa đủ "nguyên liệu" để tính:
$$
S^{\text{suyluận}}_j(q,d)=\max_{(f,f')\in\mathcal L(d)^2}\ \sigma_m(\tilde{\mathbf c}_j,\mathbf c_f)\,\sigma_m(\tilde{\mathbf c}_j,\mathbf c_{f'})\,\gamma_e\,\gamma_t(t,t_f)\,\gamma_t(t-1,t_{f'}).
$$
Cơ chế này ưu tiên tài liệu *đủ khả năng trả lời*, và là cầu nối tự nhiên sang khâu sinh.

### 5.6. (Đóng góp 2) Học tương phản với mẫu âm tạo từ tập dữ kiện, khớp theo từng luồng

Mục tiêu huấn luyện bộ truy hồi là hàm tương phản (InfoNCE) trên điểm đối sánh muộn:
$$
\mathcal L_{\text{truyhồi}}=-\log\frac{\exp(S(q,d^+)/\tau)}{\exp(S(q,d^+)/\tau)+\sum_{d^-\in\mathcal N(q)}\exp(S(q,d^-)/\tau)},
$$
với $\mathcal N(q)=\text{mẫu âm trong lô}\cup\{d^-_t,d^-_e,d^-_m,d^-_s,d^-_v\}$ — các mẫu âm tạo từ tập dữ kiện của tài liệu đúng: đổi kỳ ($d^-_t$) và đổi doanh nghiệp ($d^-_e$) huấn luyện cổng $\gamma_t,\gamma_e$; đổi khái niệm ($d^-_m$, có kiểm tra hai giá trị không trùng) buộc luồng khái niệm học tương đồng ngữ nghĩa thật; phá thang đo ($d^-_s$) và phá quan hệ số học ($d^-_v$) huấn luyện luồng độ lớn và nối với khâu kiểm chứng ở §5.7. Vì mỗi mẫu âm chỉ khác tài liệu đúng ở *một* khía cạnh, gradient dồn vào đúng thành phần tương ứng trong $S$.

**Tính chất "làm câu trả lời không còn đúng".** Gọi $\mathcal A(q)\subseteq\mathcal L(d^+)$ là tập dữ kiện trả lời $q$. Nếu phép biến đổi thay đổi khóa hoặc giá trị của *mọi* dữ kiện trong $\mathcal A(q)$ thì không thể lấy được đáp án đúng từ tài liệu đã biến đổi, nên nó không còn là mẫu dương cho $q$. Khi $\mathcal A(q)$ chỉ có một dữ kiện, đây là mẫu âm hợp lệ một cách chắc chắn; với câu hỏi nhiều dữ kiện, ta phá toàn bộ $\mathcal A(q)$ theo cấu trúc kỳ ở §5.5. Tính chất này khiến mẫu âm vừa cực khó (gần như trùng câu chữ), vừa chắc chắn hợp lệ, không cần khâu lọc tốn kém.

### 5.7. (Đóng góp 3) Huấn luyện mô hình sinh bằng phần thưởng quá trình do tập dữ kiện kiểm chứng

**Định dạng lập luận.** Ta yêu cầu mô hình sinh xuất lời giải có cấu trúc, gồm chuỗi bước $y=(s_1,\dots,s_T)$ kết thúc bằng đáp án. Mỗi bước có thể (i) *trích dẫn* một dữ kiện nguồn dưới dạng (khái niệm, doanh nghiệp, kỳ → giá trị), và/hoặc (ii) thực hiện một *phép tính* trên các giá trị đã trích để tạo giá trị trung gian. Định dạng này là chuẩn của lối suy luận có chương trình hóa và cho phép phân tích cú pháp đáng tin.

**Bộ kiểm chứng quá trình từ tập dữ kiện.** Tập dữ kiện cho phép chấm điểm từng bước một cách *tất định*, không cần gán nhãn người:
- *Phần thưởng bám nguồn* $r^{g}_i\in\{0,1\}$: giá trị mà bước $s_i$ trích có khớp một dữ kiện thật trong $\mathcal L(d)$ không (khớp khái niệm mềm theo $\sigma_m$, khớp doanh nghiệp/kỳ, dung sai giá trị)? Bước bịa con số → $0$.
- *Phần thưởng nhất quán số học* $r^{a}_i\in\{0,1\}$: giá trị trung gian có đúng bằng phép tính trên các toán hạng đã trích không; và khi các dữ kiện liên quan thuộc một đẳng thức kế toán đã biết (ví dụ Tài sản = Nợ + Vốn chủ sở hữu), đẳng thức đó có thỏa không? (Đây là nơi các đẳng thức kế toán phát huy đúng vai trò — như một bộ kiểm chứng, không phải tín hiệu xếp hạng.)

**Phần thưởng tổng hợp và tối ưu.** Với lập luận $y$,
$$
R(y)=\underbrace{R_{\text{đáp án}}(y)}_{\in\{0,1\}}+\lambda_g\cdot\frac{1}{|G(y)|}\sum_{i\in G(y)}r^{g}_i+\lambda_a\cdot\frac{1}{|A(y)|}\sum_{i\in A(y)}r^{a}_i,
$$
trong đó $G(y),A(y)$ là tập bước có trích dẫn và có phép tính. Ta tối ưu bằng GRPO: với mỗi câu hỏi, lấy mẫu một nhóm $\{y_1,\dots,y_N\}$, chuẩn hóa phần thưởng trong nhóm thành lợi thế $\hat A_i=(R(y_i)-\bar R)/\mathrm{std}(R)$, rồi cập nhật chính sách $\pi_\theta$ theo mục tiêu kẹp tỉ lệ xác suất có ràng buộc KL về mô hình tham chiếu.

**Bảo toàn ưu tiên cho độ chính xác.** Ta đặt ràng buộc $\lambda_g+\lambda_a<1$. Vì $R_{\text{đáp án}}\in\{0,1\}$ và hai số hạng quá trình nằm trong $[0,1]$, mọi lập luận *đúng đáp án* luôn có phần thưởng cao hơn mọi lập luận *sai đáp án*, bất kể quá trình. Nhờ đó độ chính xác là ưu tiên tuyệt đối (theo thứ tự từ điển), còn phần thưởng quá trình chỉ phân biệt giữa các lập luận cùng độ chính xác — đẩy mô hình về lập luận vừa đúng vừa có căn cứ. Đây là cách diễn đạt nguyên tắc "không đánh đổi đúng số lấy nhất quán" bằng đúng một bất đẳng thức, tránh thêm tham số thừa.

**Vì sao giải được vấn đề.** Phần thưởng theo đáp-án-cuối đơn thuần tưởng thưởng cả những lời giải đúng nhờ may rủi hoặc nhờ các bước sai triệt tiêu nhau; phần thưởng quá trình từ tập dữ kiện phạt ngay các bước bám nguồn sai hoặc tính sai, nên chính sách bị đẩy về suy luận thực sự có căn cứ. Khác với bộ thưởng theo từng bước phải gán nhãn, bộ kiểm chứng ở đây *tất định và miễn phí*; khác với thưởng trung thực ở mức văn bản, ở đây kiểm tra *giá trị số* và *phép tính*.

### 5.8. Mối gắn kết ba chiều

Ba đóng góp là ba lần áp dụng cùng một nguyên lý *lập chỉ mục theo dữ kiện*, trên cùng một tập dữ kiện.
- **Đóng góp 1 → Đóng góp 3:** tập dữ kiện và các tài liệu truy hồi được chính là *bộ kiểm chứng* và *nguồn trích dẫn* cho lập luận; "suy luận theo dữ kiện" soi gương "độ liên quan theo dữ kiện".
- **Đóng góp 2 ↔ Đóng góp 3:** chính nguyên lý "biến đổi một dữ kiện làm mất hiệu lực đáp án" của Đóng góp 2 định nghĩa thế nào là một bước "bám nguồn sai" ở cấp lập luận; các phép biến đổi đó còn sinh ra *lập luận đối chứng* (trích dẫn giá trị đã bị đổi, hoặc phép tính bị phá) làm mẫu phủ định chắc chắn để ổn định huấn luyện hoặc tạo cặp ưu tiên.
- **Đóng góp 2 ↔ Đóng góp 1:** mỗi loại mẫu âm giám sát đúng một cổng/luồng; ngược lại chính cơ chế tách luồng có cổng làm cho mẫu âm vừa cực khó vừa học được.

Sự cần đến nhau theo cả ba chiều cho thấy đây là một lời giải thống nhất ở cấp hệ thống, không phải ba kỹ thuật ghép rời.

---

## 6. Thiết kế đánh giá

**Dữ liệu.** T²-RAGBench gồm FinQA, ConvFinQA, TAT-DQA (tổng 23.088 câu hỏi trên 7.318 báo cáo), đánh giá trong điều kiện không cho sẵn ngữ cảnh, cho cả truy hồi và sinh. Một bộ dữ liệu tài chính tiếng Việt theo Chuẩn mực Kế toán Việt Nam được xây dựng như một nghiên cứu trường hợp về khả năng chuyển sang ngôn ngữ khác.

**Các phương pháp so sánh.** *Truy hồi:* BM25; lai BM25; truy hồi vector (BGE-M3, multilingual-e5-large, text-embedding-3-large); ColBERTv2; bộ chấm điểm lại; HyDE; truy hồi có tính đến đại lượng; và truy hồi vector kết hợp lọc cứng theo doanh nghiệp/năm rồi chấm điểm lại. *Sinh:* nhắc trực tiếp (few-shot), suy luận có chương trình hóa, học tăng cường theo đáp-án-cuối, và bộ thưởng theo từng bước có gán nhãn — để cô lập giá trị của phần thưởng quá trình tất định.

**Độ đo.** Truy hồi: MRR@3, Recall@{1,3,5}, nDCG@3. Sinh: độ chính xác con số, tỉ lệ vi phạm ràng buộc, và một độ đo *trung thực số học* (tỉ lệ bước bám nguồn đúng và nhất quán). Hệ thống: độ trễ và thông lượng.

**Phân tích cơ chế.** (i) Bộ dò tuyến tính và độ phân tán để kiểm chứng §4; (ii) sự đảo dấu của chênh lệch tương đồng đúng−sai (từ âm sang dương); (iii) hiệu năng theo nhóm câu hỏi, đặc biệt nhóm *cùng doanh nghiệp và cùng năm* — nơi phải chọn đúng chỉ tiêu/ô giữa hàng chục con số, và là nơi lọc định danh đơn thuần không giải quyết được; (iv) mức suy giảm khi đưa nhiễu vào bộ phân tích câu hỏi (đo độ bền của dự phòng nới lỏng); (v) chất lượng trích dữ kiện, trọng tâm là *tỉ lệ lấy được đúng dữ kiện chứa đáp án*, cùng khoảng cách giữa tập dữ kiện tự động và tập dữ kiện lý tưởng; (vi) tỉ lệ "đúng đáp án nhưng sai lập luận" giảm bao nhiêu nhờ phần thưởng quá trình, so với học theo đáp-án-cuối.

**Nghiên cứu loại bỏ thành phần.** Gỡ riêng từng thành phần: cổng doanh nghiệp – thời gian; luồng độ lớn tách riêng; so khớp cấp dữ kiện so với một-vector; tìm kiếm có mặt nạ; dự phòng nới lỏng; từng loại mẫu âm (đặc biệt là đổi khái niệm); thành phần suy luận nhiều bước; và từng số hạng của phần thưởng quá trình (bám nguồn, nhất quán số học) cùng ràng buộc $\lambda_g+\lambda_a<1$.

---

## 7. Đóng góp kỳ vọng và Ý nghĩa

LEDGER hướng tới: (i) một bộ truy hồi tài chính vượt các phương pháp so sánh mạnh, cải thiện rõ nhất ở những nhóm câu hỏi khó mà tương đồng ngữ nghĩa và lọc định danh đều không xử lý được; (ii) một cách sinh mẫu âm tổng quát — biến đổi dữ kiện chứa đáp án để thu mẫu âm chắc chắn hợp lệ, khớp theo từng luồng — áp dụng được cho mọi kho tài liệu có phiên bản hoặc có cấu trúc; (iii) một phần thưởng quá trình *tất định, ký hiệu, miễn gán nhãn* cho suy luận số, biến tập dữ kiện thành nguồn giám sát từng bước và giảm hiện tượng "đúng đáp án nhưng sai lập luận". Quan trọng hơn, ba đóng góp cho thấy một nguyên lý duy nhất — *lập chỉ mục theo dữ kiện* — có thể chi phối nhất quán cả truy hồi, huấn luyện truy hồi và huấn luyện sinh, mang lại một khung hỏi–đáp tài chính gắn kết và có nền tảng lý thuyết trên một bộ dữ liệu còn nhiều dư địa.

---

## 8. Phạm vi và Giới hạn

LEDGER giả định có thể phân tích bảng thành các dữ kiện có cấu trúc; với bảng nhiều tầng phức tạp, hệ thống dựa vào cơ chế lui ở cấp dòng văn bản, và hiệu năng phụ thuộc vào tỉ lệ lấy được đúng dữ kiện chứa đáp án — một yếu tố được đo lường tường minh. Bộ phân tích câu hỏi có thể sai; thiết kế cổng mềm và dự phòng nới lỏng giúp hiệu năng suy giảm từ từ. Bộ kiểm chứng quá trình bao phủ các phép số học và các đẳng thức kế toán phổ biến; những câu hỏi định tính hoặc suy luận ngoài phạm vi số sẽ chỉ nhận phần thưởng theo đáp án cuối. Phân tích ở §4 là mô hình lý tưởng hóa với giả định nêu rõ, đóng vai trò định hướng và được kiểm chứng bằng đo đạc.

---

## Tài liệu tham khảo (chọn lọc)

- T²-RAGBench: Text-and-Table Benchmark for Retrieval-Augmented Generation. EACL 2026.
- Dense Retrievers Can Fail on Simple Queries: Revealing the Granularity Dilemma of Embeddings. Findings of EMNLP 2025.
- Revealing the Numeracy Gap: An Empirical Investigation of Text Embedding Models. Findings of EACL 2026.
- Numbers Matter! Bringing Quantity-awareness to Retrieval Systems. Findings of EMNLP 2024.
- SoarGraph / Doc2SoarGraph: Numerical Reasoning over Financial Table-Text Data via Semantic-Oriented Hierarchical Graphs. WWW 2023.
- Structure First, Reason Next: Knowledge Graph for Numerical Reasoning in Financial Documents. 2026.
- ColBERT / ColBERTv2: Efficient and Effective Retrieval via Late Interaction. SIGIR 2020 / NAACL 2022.
- xVal: A Continuous Numerical Tokenization for Scientific Language Models. 2023.
- RocketQA; NV-Retriever: khai thác mẫu âm khó và xử lý mẫu âm giả cho truy hồi vector.
- DeepSeek-R1; GRPO: Reinforcement Learning with Verifiable Rewards cho suy luận.
- Let's Verify Step by Step; Math-Shepherd: Process Reward Models và nút thắt gán nhãn.
- Beyond Correctness: Rewarding Faithful Reasoning in Retrieval-Augmented Generation. 2025.
- FinQA (EMNLP 2021); TAT-QA (ACL 2021); ConvFinQA (EMNLP 2022).
- The Information Bottleneck Method. 1999.
