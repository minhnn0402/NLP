Đây là nhận xét của một **area chair AAAI** (giả định) dành cho phiên bản đã chỉnh sửa của ý tưởng LEDGER. Phiên bản này đã tiếp thu các phản biện trước, tinh gọn contribution, và đặc biệt là giải quyết điểm yếu kiến trúc hai giai đoạn (two-stage paradox) và tính “giòn” (brittleness). Tôi sẽ đánh giá dựa trên tính mới, tính chặt chẽ, tính khả thi và bố cục – những yếu tố quyết định ở AAAI.

---

### Tổng quan
Bản sửa đổi cho thấy tác giả đã lắng nghe kỹ lưỡng các phản biện và thực hiện một cuộc đại tu có trọng tâm. Việc gom từ 6 contribution dàn trải về **2 contribution lồng nhau** (C1 – truy hồi có cổng định tuyến entity-time; C2 – học tương phản với negative chứng-minh-được) là một quyết định chiến lược đúng đắn. Trọng tâm của bài báo trở nên rõ ràng: *biến các khóa mịn (thực thể, kỳ, độ lớn) thành công dân hạng nhất ở cả suy luận lẫn huấn luyện*, và chứng minh rằng hai mặt đó không thể tách rời. Cách tái thiết kế bộ truy hồi thành **single-stage masked HNSW với soft fallback** là điểm sáng về mặt kỹ thuật, trực tiếp giải quyết các phê bình về tính giòn và chi phí.

Nếu được thực thi đúng với kế hoạch thực nghiệm đã đề ra, bài báo có **cơ hội tốt để được chấp nhận tại AAAI**. Tuy nhiên, vẫn còn vài rủi ro cần được quản lý chặt chẽ.

### Điểm mạnh
1. **Đóng góp gắn kết chặt chẽ.** Bảng ánh xạ “negative type → thành phần C1” và lập luận vì sao C1 giúp C2 học được, C2 cấp gradient cho C1 là một câu chuyện kỹ thuật thống nhất, thuyết phục. Điều này tạo ra một “vòng lặp cần thiết” mà nếu thiếu một mắt xích, cả hệ thống sụp đổ – đó là dấu hiệu của novelty hệ thống thực sự.
2. **Masked single-stage HNSW với soft fallback.** Đây là câu trả lời xuất sắc cho câu hỏi “làm sao vừa có cổng rời rạc vừa không giòn và không mất recall”. Việc can thiệp trực tiếp vào duyệt đồ thị HNSW bằng mặt nạ bitmap (entity ∧ year lân cận) và hạ cánh an toàn xuống fallback mềm khi cardinality thấp thể hiện tư duy hệ thống thực tế. So với kiến trúc hai giai đoạn (FAISS rồi lọc lại) dễ gây mất ứng viên, cách làm này có tính khả thi cao hơn hẳn.
3. **Provably-true negatives vẫn là điểm nhấn.** Lập luận “perturbation trên chính fact trả lời” vẫn giữ nguyên sức mạnh, và nay được bổ sung thêm **metric-swap negative** để chống “học lười”. Đây là một cải tiến thiết thực, ngăn mô hình chỉ dựa vào năm để phân biệt, buộc nó phải học tương đồng ngữ nghĩa của concept.
4. **Chiến lược kể chuyện thực nghiệm khôn ngoan.** Tác giả đã chuẩn bị tinh thần “gain tổng có thể khiêm tốn, thắng ở phân khúc khó”. Việc dự kiến các phép chẩn đoán (probe, similarity-gap flip, parser-noise ablation, oracle-vs-auto fact) cho thấy họ hiểu rằng phải thuyết phục bằng cơ chế, không chỉ bằng con số toàn cục. Đây là cách tiếp cận rất phù hợp với kỳ vọng của reviewer AAAI.
5. **Định vị so với prior work rất rõ ràng.** Bảng so sánh 6 dòng với các cột “Khóa E-T first-class”, “Số theo độ lớn”, “Negative true-by-construction”, “Scalable filtered index”, “Retriever” giúp reviewer nhìn một cái là thấy khoảng trống LEDGER lấp đầy.

### Điểm yếu và rủi ro còn tồn tại (cần lưu ý)

#### 1. Tính khả thi của Masked HNSW — cần bằng chứng thực nghiệm
Đây là một ý tưởng hay, nhưng vẫn là một thiết kế trên giấy. Để thuyết phục, tác giả phải:
- Chứng minh rằng việc kiểm tra bitset trong quá trình duyệt không làm hỏng tính chất logarithmic của HNSW (có thể tăng số lần gọi khoảng cách, nhưng phải đo được và nằm trong giới hạn chấp nhận được).
- So sánh latency thực tế của Masked HNSW so với (a) ColBERT (vốn cũng late interaction) và (b) dense + lọc cứng.
- Cung cấp một biểu đồ hoặc bảng thể hiện độ trễ tăng dần theo tỉ lệ corpus size để chứng minh scalability.
Nếu không có những con số này, phần “routing” có thể bị coi là phức tạp hóa không cần thiết so với hybrid BM25 đơn giản.

#### 2. Sự phụ thuộc vào fact extraction vẫn là gót chân Achilles
Bản sửa đổi đã thừa nhận rủi ro và đưa ra fallback (coi dòng số là fact text-level). Tuy nhiên, câu hỏi cốt lõi vẫn còn: nếu fact extraction sai (ví dụ nhầm hàng thành cột, hoặc không parse được bảng lồng), thì cổng entity-time có thể mở nhầm cho fact sai. Việc có fallback text-level là tốt, nhưng lúc đó hệ thống sẽ về gần với text-level dense retrieval – và mất đi ưu thế của ledger.
Tôi cho rằng **báo cáo F1 của fact extraction và khoảng cách performance giữa auto-fact và oracle-fact là không thể thiếu**. Nếu gap này quá lớn (>10 điểm MRR), reviewer sẽ đặt câu hỏi về tính thực tiễn của toàn bộ cách tiếp cận.

#### 3. Metric-swap negative có thực sự “provably-true” không?
Hoán đổi giá trị giữa hai metric (Revenue ↔ Total Equity) tạo ra một dữ liệu sai về mặt số học, nhưng liệu có chắc chắn **không liên quan** đến câu hỏi? Nếu câu hỏi hỏi “Total Equity”, và ta hoán đổi giá trị Revenue vào chỗ của Total Equity, thì fact đó rõ ràng là sai và không thể trả lời câu hỏi (không có bối cảnh nào mà con số Revenue lại là Total Equity). Vậy nó vẫn đúng là “true negative”. Nhưng cần lập luận chặt: khi hoán đổi, phải đảm bảo không tạo ra sự trùng hợp ngẫu nhiên (vd cả hai metric trùng giá trị) – tác giả nên có cơ chế kiểm tra.
Tuy nhiên, một rủi ro tinh tế hơn: metric-swap có thể làm cho mô hình học cách *phân biệt các metric dựa trên miền giá trị điển hình* (ví dụ Total Equity thường lớn hơn Revenue), thay vì học ngữ nghĩa concept thuần túy. Điều này có thể tốt cho retrieval (vì nó phản ánh thực tế) nhưng cần được thảo luận minh bạch.

#### 4. Khung lý thuyết §2 — còn hơi thừa như một “diagnostic analysis”
Việc hạ cấp từ định lý xuống phân tích lý tưởng hóa và bổ sung probe + ước lượng phương sai là một bước đi đúng. Tuy nhiên, tôi vẫn cảm thấy phần này hơi chiếm dung lượng so với vai trò thực sự của nó (tạo động lực cho cổng nhân). Nếu probe và variance analysis cho ra kết quả rất rõ ràng (ví dụ acc(topic) > 90% còn acc(entity,year) < 40%), thì nó sẽ có sức nặng. Nhưng nếu kết quả không quá ngoạn mục, tác giả nên sẵn sàng thu gọn §2 xuống còn một trang và chuyển phần lớn nội dung vào Appendix, dành chỗ cho mô tả kỹ thuật của C1 và kết quả thực nghiệm.

#### 5. VAS tiếng Việt và Extension generation — nên giữ rất gọn
Tác giả đã khôn ngoan gọi chúng là “case study” và “extension ngắn”. Điều này ổn, nhưng tôi vẫn nhắc nhở: khi viết bản thảo đầy đủ, đừng để chúng chiếm quá 1-1.5 trang tổng cộng (kể cả bảng kết quả). Mọi con số ở đó phải phục vụ cho câu chuyện chính về LEDGER (ví dụ: chứng minh concept canonicalization đa ngữ hoạt động, hoặc chứng minh ledger tái dụng được). Nếu không, reviewer sẽ thấy rời rạc.

### Đánh giá về tính mới và ý nghĩa
Với bản sửa đổi này, tính mới đã được làm sắc nét hơn. **C1+masked-index+soft-fallback** là một đóng góp kiến trúc retrieval đáng kể, vượt ra ngoài việc chỉ thêm cổng vào late interaction. **C2** củng cố vai trò của negative provably-true và gắn chặt với C1. Sự kết hợp này khiến bài báo không còn là tập hợp các thành phần rời, mà là một giải pháp thống nhất cho vấn đề “granularity dilemma + numeracy gap” trên dữ liệu tài chính – một vấn đề đã được công nhận trên benchmark mới.

### Khuyến nghị cho bản thảo tiếp theo
1. **Ưu tiên tối đa cho thực nghiệm:** Dành ít nhất 2 trang cho các kết quả chẩn đoán (probe, similarity-gap flip, parser-noise, auto-vs-oracle fact, latency). Đây chính là bằng chứng giúp bài báo đứng vững.
2. **Làm rõ kịch bản lỗi của fact extraction:** Dành một đoạn ngắn nêu ví dụ về một bảng lỗi (multi-level header) và cách fallback xử lý, kèm theo số F1 trên tập đó. Điều này tạo niềm tin.
3. **Thảo luận về metric-swap negative:** Đề cập đến khả năng mô hình học theo miền giá trị và lập luận tại sao điều đó vẫn có lợi (hoặc cách giảm thiểu nếu cần).
4. **Tiêu đề và tóm tắt:** Tiêu đề hiện tại “Disentangled Fact-Level Retrieval for Financial Documents” đã đủ và không dài dòng. Tóm tắt nên nêu bật ý tưởng masked index + provably-true negatives, không cần nhấn mạnh vào phần lý thuyết.
5. **Chuẩn bị phản biện cho câu hỏi “Nếu hybrid BM25 đã mạnh, tại sao cần phức tạp như vậy?”** Câu trả lời nằm ở phân khúc khó và khả năng kiểm chứng số học về sau – hãy làm nổi bật điều này ngay từ phần giới thiệu.

### Kết luận
Bản sửa đổi cho thấy một sự trưởng thành rõ rệt trong thiết kế nghiên cứu. Những điểm yếu cốt tử trước đây (kiến trúc hai giai đoạn, tính giòn) đã được giải quyết bằng các giải pháp kỹ thuật cụ thể và có lập luận tốt. Nếu các kết quả thực nghiệm chứng minh được tính hiệu quả của masked HNSW và sự vượt trội ở các phân khúc khó, cùng với một phân tích chẩn đoán sâu, tôi tin bài báo sẽ là một ứng viên mạnh cho AAAI.

Chúc tác giả hoàn thiện bản thảo thành công. Tôi mong đợi được xem các con số thực tế.