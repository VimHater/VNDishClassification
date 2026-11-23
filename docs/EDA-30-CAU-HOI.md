# 30 câu hỏi EDA trọng yếu

Dành cho **Người 2 (EDA)**, đọc kèm [`GIAO-VIEC-N2.md`](GIAO-VIEC-N2.md). Mỗi câu ra **một hình hoặc một bảng**, và ghi rõ kết quả dùng
cho quyết định hay thực nghiệm nào. Người 3 dùng các kết quả này để dự đoán và giải thích kết quả E1, E2, E7, E3.

**Bộ dữ liệu:** VNFood103 (Kaggle `meowluvmatcha/vnfood-30-100`, 103 món, 49.223 ảnh), Người 1 làm sạch thành `vnfood103-clean`.
Bộ gốc ghép từ 3 nguồn: **30vnfoods** (25.136 ảnh, 30 món), **100vnfoods** (20.700 ảnh, 100 món, 70 món chỉ có nguồn này), **vinafood21** (3.387 ảnh, 3 món: bánh hỏi, bánh mì chảo, bò lá lốt).
Số liệu trong ngoặc là số đã kiểm tra trực tiếp trên file zip (23/11/2026). Nhiều câu cần biết **ảnh thuộc bộ nào**: lấy từ `merge_manifest.csv`
(có trong `vnfood-prep`, cột `source`, `orig_split`), nối với `splits.csv` theo **tên file**:
`src = df.path.str.split("/").str[-1].map(dict(zip(man.path.str.split("/").str[-1], man.source)))`.
Các nhóm dễ nhầm: món nước sợi; món kho/sốt nâu (cá kho tộ, thịt kho tàu, bò kho); và các cặp **gần như cùng một món nhưng khác nguồn**
(bò lá lốt – bò nướng lá lốt, bánh mì chảo – bò né, lươn om chuối đậu – ba ba nấu chuối đậu).

**Ký hiệu:** ★ = làm trước (trả lời thẳng câu hỏi của thầy, hoặc phát hiện vấn đề dữ liệu).
**Nguồn:** `prep` = Kaggle Dataset `vnfood-prep` (Người 1); `gốc` = Kaggle Dataset `meowluvmatcha/vnfood-30-100` trước khi làm sạch (cần cho câu 3, 6, 8).

**Nguyên tắc:** các phân tích dựa trên mô hình (1-NN, zero-shot…) chỉ chạy trên **train và val**, không chạy trên test.
Ngoại lệ duy nhất là câu 8: câu này đo đúng mức rò rỉ của test, không dùng để chọn tham số.

## Thứ tự ưu tiên

1. **Chất lượng dữ liệu** (vấn đề lớn thì báo Người 1 và Người 3): ★ 4, 5, 7, 8, 9, 10, 11.
2. **Bằng chứng cho câu hỏi của thầy** (vì sao riêng HOG/màu/CLIP không đủ): ★ 17, 20, 21, 24, 25, 26.
3. **Còn lại**, làm theo thứ tự trong file.

## Code dùng chung

Notebook Kaggle mới (GPU không bắt buộc, trừ câu 28–30), **Add Input** `vnfood-prep`:

```python
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
PREP = next(Path("/kaggle/input").rglob("prep_manifest.json")).parent
df = pd.read_csv(PREP / "splits.csv")                       # mỗi ảnh một dòng; chỉ dùng split train/val/test
meta = json.load(open(PREP / "classes.json", encoding="utf-8"))
NAME = [meta["info"][l]["vi"] for l in meta["labels"]]
IMGS = np.load(PREP / "images.npy", mmap_mode="r")         # (N,144,144,3), cùng thứ tự với df
FEATS = {k: np.load(PREP / f"feat_{k}.npy") for k in ["color", "hog", "clip", "siglip"]}
man = pd.read_csv(PREP / "merge_manifest.csv")             # nguồn từng ảnh (bộ gộp)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
y = df.y.values
TR, VA = (df.split == "train").values, (df.split == "val").values
C = len(NAME)

def prep(X):
    """Chuẩn hóa theo train rồi chuẩn L2, để so bằng cosine."""
    Z = StandardScaler().fit(X[TR]).transform(X)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)
Z = {k: prep(X) for k, X in FEATS.items()}

def knn_labels(Zk, query_mask, k=10):
    """Nhãn của k ảnh TRAIN gần nhất (cosine) với mỗi ảnh trong query_mask -> (Q, k)."""
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Zk[TR])
    return y[TR][nn.kneighbors(Zk[query_mask], return_distance=False)]

def class_similarity(Zk):
    """Cosine giữa vector trung bình của từng món trên train -> (C, C)."""
    M = np.stack([Zk[TR & (y == c)].mean(0) for c in range(C)])
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return M @ M.T

def save(name):
    plt.tight_layout(); plt.savefig(f"/kaggle/working/{name}.png", dpi=130, bbox_inches="tight"); plt.show()
```

---

## A. Dữ liệu có đúng và đủ không

**1. Mỗi món có bao nhiêu ảnh ở mỗi tập, lệch lớp bao nhiêu lần?**
- Làm: `pd.crosstab(df.label, df.split)` trên ảnh hợp lệ; tỉ lệ món nhiều nhất / ít nhất. (Bộ gốc trước khử trùng: nậm pịa 153 ảnh, bún bò Huế 1.767 → lệch 11,5 lần; 70 món mới chỉ khoảng 150–280 ảnh.)
- Ra: biểu đồ cột ngang chồng train/val/test, sắp theo tổng số ảnh (đã có `eda/class_distribution.png`) + bảng số.
- Dẫn tới: báo cáo **macro-F1**, không chỉ accuracy. Thực nghiệm nhỏ: LogReg trên CLIP có và không có `class_weight="balanced"`, so macro-F1 trên **val**.

**2. Tỉ lệ train/val/test có giống nhau giữa các món không?**
- Làm: với mỗi món, % ảnh ở train, val, test; độ lệch chuẩn của % test giữa các món.
- Ra: biểu đồ cột chồng 100% theo món.
- Dẫn tới: món nào có quá ít ảnh test (< 80) thì độ chính xác của món đó dao động mạnh, đừng kết luận từ một món lẻ.

**3. Mỗi nguồn đóng góp bao nhiêu ảnh, cho món nào; qua từng bước làm sạch còn bao nhiêu?** — nguồn `gốc` + `prep`
- Làm: bảng số ảnh theo nguồn ở 3 giai đoạn: bộ gốc, sau làm sạch (`vnfood103-clean`), sau Phần A (chỉ tính train/val/test).
  Bộ gốc: 49.223 ảnh (30vnfoods 25.136 — khớp đủ bộ 30VNFoods gốc; 100vnfoods 20.700; vinafood21 3.387); có 571 ảnh xám, 565 thuộc vinafood21.
- Ra: bảng nguồn × giai đoạn + bảng món × nguồn.
- Dẫn tới: giải thích vì sao không so trực tiếp với mốc 77,54% của bài 30VNFoods (khác số món, khác tập test).

**4. ★ Mỗi bước làm sạch loại bao nhiêu ảnh, theo từng món?**
- Làm: đếm `split ∈ {bad, conflict, dup}` theo món từ `splits.csv` (`dup` = ảnh trùng, mỗi nhóm chỉ giữ 1 ảnh lớn nhất).
  Tách thêm theo nguồn: kỳ vọng `dup` rơi nhiều vào 100vnfoods ở 30 món chung (nhiều ảnh là bản thu nhỏ của 30VNFoods).
- Ra: bảng món × lý do loại, sắp theo tổng.
- Dẫn tới: món nào mất > 10% ảnh thì báo Người 1 xem lại (có thể ngưỡng pHash đang gom nhầm).

**5. ★ Còn đoán được "ảnh này của bộ nào" không?** — `prep` + `merge_manifest.csv`
- Làm:
  (a) **Trong cùng một món:** chỉ dùng 30 món có cả 30vnfoods và 100vnfoods. Huấn luyện LogReg đoán **nguồn** từ đặc trưng màu, HOG, CLIP (train → val),
      cân bằng số ảnh hai nguồn. Gần 50% = không phân biệt được (tốt); xa 50% = vẫn còn dấu hiệu nguồn.
  (b) **Thống kê đơn giản theo nguồn:** so phân bố `width`, `height`, `kb`, `brightness`, `sharpness` giữa các nguồn.
      (Bộ gốc: 100vnfoods trung vị 386×290 px, 40% ảnh có cạnh ngắn < 224 px; 30vnfoods 1.000×720, chỉ 1,4%.)
  (c) Các cặp ảnh trùng (pHash) giữa hai nguồn ở 30 món chung.
- Ra: bảng độ chính xác đoán nguồn theo từng đặc trưng + boxplot thống kê theo nguồn.
- Dẫn tới: 70 món chỉ có ảnh 100vnfoods (nhỏ, mờ hơn) → mô hình có thể học "ảnh mờ → một trong 70 món". Nếu (a) cao thì báo Người 3 đọc kỹ ô B6b; ghi nhận làm hạn chế.

## B. Dữ liệu có sạch không

**6. Có bao nhiêu ảnh trùng y hệt và gần trùng?** — nguồn `gốc` + `prep`
- Làm: trùng y hệt bằng MD5 của file (bộ gốc: 4.997 nhóm, 10.093 ảnh; 52 nhóm khác nhãn); gần trùng bằng cột `group` (pHash) trong `splits.csv`.
- Ra: bảng (MD5 so với pHash; trong cùng món / khác món; cùng tập / khác tập).
- Dẫn tới: cho thấy pHash bắt được nhiều hơn MD5 bao nhiêu (ảnh đổi cỡ, nén lại).

**7. ★ Ngưỡng pHash = 4 có hợp lý không?**
- Làm: từ cột `phash` (hex 64 bit), đếm số cặp ảnh ở mỗi khoảng cách Hamming 0–12 (chỉ trong cùng món là đủ nhanh). Mỗi mức khoảng cách xem 3 cặp ví dụ.
- Ra: đường cong "số cặp theo khoảng cách" + lưới ví dụ (mỗi hàng một mức khoảng cách).
- Dẫn tới: mức nào ảnh vẫn còn giống hệt, mức nào bắt đầu khác. Nếu nên đổi `max_dist` thì báo Người 1.

**8. ★ Bao nhiêu % ảnh test có bản trùng ở train/val, và nó làm điểm số phồng lên bao nhiêu?**
- Làm: dùng train/val/test **chia sẵn của bộ gốc** (cột `orig_split` trong `merge_manifest.csv`), đếm ảnh test thuộc nhóm `group` có ảnh train.
  (Kiểm tra nhanh bằng dHash: 3.694 / 14.767 ảnh val/test = 25% có bản gần trùng trong train; 100vnfoods 31%, 30vnfoods 20%, vinafood21 26%.)
  **Thực nghiệm:** huấn luyện 1-NN trên CLIP với train gốc. So accuracy trên **test gốc** (còn ảnh trùng) với **test đã làm sạch**.
- Ra: bảng (số ảnh rò rỉ, accuracy hai cách) + 6 cặp ví dụ train ↔ test.
- Dẫn tới: một con số cụ thể cho thấy vì sao không dùng cách chia của bộ gốc, và vì sao nhóm phải khử trùng rồi tự chia lại.

**9. ★ Ảnh trùng khác nhãn rơi vào những cặp món nào?**
- Làm: từ các nhóm `conflict`, đếm cặp nhãn. (Bộ gốc có 52 nhóm trùng y hệt khác nhãn, ví dụ cùng một ảnh vừa là bánh canh, bún riêu vừa là hủ tiếu; bánh bèo / bánh bột lọc / bánh đúc.)
- Ra: ma trận "nhãn A ↔ nhãn B" + lưới ví dụ.
- Dẫn tới: danh sách cặp món mà **chính người gán nhãn** cũng lẫn lộn. Đây là dự đoán sớm cho cặp nhầm ở E1/E2/E7.

**10. ★ Bao nhiêu % ảnh bị sai nhãn, không phải đồ ăn, có nhiều món, hoặc là ảnh ghép?**
- Làm: mỗi món xem 20 ảnh train ngẫu nhiên (`random_state=42`), đếm bằng tay 4 loại lỗi.
- Ra: bảng tỉ lệ lỗi theo món + 1 lưới ví dụ mỗi loại.
- Dẫn tới: ước lượng trần độ chính xác (nhãn sai 5% thì khó vượt 95%); món nhiều lỗi thì báo Người 1.

**11. ★ Bao nhiêu ảnh có chữ, logo, watermark hoặc là ảnh bìa video?**
- Làm: xem 20 ảnh ngẫu nhiên mỗi món, ưu tiên 70 món của 100vnfoods (nhiều ảnh bán hàng, có chữ, giá, số điện thoại). Đếm ảnh có chữ, và ảnh mà chữ **ghi luôn tên món**.
- Ra: bảng + lưới ví dụ.
- Dẫn tới: CLIP đọc được chữ trong ảnh. Ảnh có ghi "phở" thì CLIP "ăn gian" nhờ đọc chữ chứ không nhìn món. Nếu nhiều, đo riêng độ chính xác trên ảnh có chữ và không chữ.

**12. Có cặp món nào thực chất là một món không?**
- Làm: các cặp nghi trùng nghĩa: bò lá lốt (vinafood21) – bò nướng lá lốt (100vnfoods), bánh mì chảo – bò né, lươn om chuối đậu – ba ba nấu chuối đậu,
  canh khổ qua – canh chua, tàu hũ nhồi thịt – cá sốt cà chua. Mỗi cặp: lưới 8 + 8 ảnh train; cosine giữa tâm hai món theo CLIP (`class_similarity`);
  tỉ lệ 1-NN CLIP trên val nhầm qua lại giữa hai món.
- Ra: lưới ảnh từng cặp + bảng (cosine tâm, tỉ lệ nhầm).
- Dẫn tới: cặp nào gần như không phân biệt được bằng mắt thì báo cáo như **nhập nhằng nhãn** của dữ liệu, không phải lỗi mô hình;
  nếu nhóm quyết định gộp cặp đó thành một món thì báo Người 1 trước khi chốt `vnfood-prep`.

## C. Ảnh trông thế nào

**13. Ảnh lớn cỡ nào, bao nhiêu ảnh nhỏ hơn đầu vào của CLIP (224 px)?**
- Làm: `width`, `height` trong `splits.csv`. (Bộ gốc: 100vnfoods 40,1% ảnh có cạnh ngắn < 224 px; 30vnfoods 1,4%; vinafood21 1,7%.) Tách theo nguồn.
- Ra: histogram cạnh ngắn có vạch 224 (đã có `eda/image_sizes.png`) + bảng theo món.
- Dẫn tới: ảnh nhỏ phải phóng to và bị mờ. Thực nghiệm: 1-NN CLIP trên val theo 3 nhóm kích thước.

**14. Cắt giữa (center crop) làm mất bao nhiêu phần ảnh?**
- Làm: tỉ lệ rộng/cao; phần diện tích bị cắt = `1 − min(w,h)/max(w,h)`. Vẽ khung cắt lên 8 ảnh dài nhất.
- Ra: histogram phần bị cắt + lưới ảnh có khung.
- Dẫn tới: nếu nhiều ảnh mất món khi cắt giữa thì cân nhắc thêm viền (padding) thay vì cắt. Hiện cả notebook và CLIP đều cắt giữa.

**15. Ảnh xám trong bộ gốc là gì?** — *đã làm khi chuẩn hóa, chỉ cần trình bày lại* (nguồn `gốc`)
- Kết quả: 571 ảnh xám, 565 thuộc vinafood21 (≈ 1/6 ảnh của 3 món đó), nên gần như chắc là ảnh tăng cường dữ liệu chứ không phải ảnh gốc:
  phần lớn có bản ảnh màu trùng. Giữ lại thì "ảnh xám → bánh hỏi / bánh mì chảo / bò lá lốt" thành lối tắt, nên bước làm sạch bỏ hết.
- Ra: lưới 6 cặp ảnh màu ↔ bản xám + bảng tỉ lệ ảnh xám theo nguồn và theo món.
- Dẫn tới: lý do bỏ ảnh xám.

**16. Độ sáng, độ nét có khác nhau theo món hay giữa train và test không?**
- Làm: `brightness`, `sharpness` trong `splits.csv`; boxplot theo món. So phân bố train và test bằng `scipy.stats.ks_2samp`.
- Ra: 2 boxplot + bảng p-value của kiểm định KS.
- Dẫn tới: món nào tối hoặc mờ một cách hệ thống thì mô hình có thể dựa vào độ sáng như một lối tắt. Train khác test thì có lệch phân phối.

## D. Màu thấy gì

**17. ★ Màu của các món chồng lấn tới đâu?**
- Làm: histogram hue lấy từ đặc trưng màu có sẵn: `H = FEATS["color"][:, :512].reshape(-1, 8, 8, 8).sum((2, 3))` (8 ô hue).
  Lấy trung bình theo món. Vẽ chồng các món nước sợi; vẽ thêm mỗi món một điểm (hue trội, độ bão hòa trung bình).
- Ra: histogram chồng nhóm món sợi + biểu đồ điểm cho mọi món.
- Dẫn tới: hình minh họa "bún bò và bún riêu cùng đỏ cam, phở và hủ tiếu cùng nước trong" → **vì sao riêng màu không đủ**.

**18. Nếu chỉ dùng màu, cặp món nào gần nhau nhất?**
- Làm: `class_similarity(Z["color"])`; liệt kê 10 cặp gần nhất.
- Ra: heatmap C × C + bảng 10 cặp.
- Dẫn tới: dự đoán cặp nhầm của E1 (màu). Sau khi có kết quả E1 thì đối chiếu xem đúng không.

**19. Chỉ dùng màu thì món nào nhận được, món nào không?**
- Làm: độ thuần kNN theo món: `(knn_labels(Z["color"], VA) == y[VA][:, None]).mean(1)`, lấy trung bình theo món.
- Ra: biểu đồ cột theo món.
- Dẫn tới: món nào màu đã gần đủ, món nào màu vô dụng; dùng lại ở câu 24.

## E. HOG thấy gì

**20. ★ HOG "nhìn" thấy gì ở một tô phở?**
- Làm: mỗi món nước sợi 1 ảnh; `skimage.feature.hog(gray128, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), visualize=True)`,
  vẽ ảnh gốc cạnh ảnh HOG.
- Ra: lưới 2 hàng (ảnh gốc / HOG) × các món sợi.
- Dẫn tới: HOG chỉ còn các vòng cạnh của cái tô, mất màu nước dùng và loại sợi → **vì sao riêng HOG không đủ**.

**21. ★ Cặp món gần nhau theo HOG có giống cặp gần nhau theo màu không?**
- Làm: `class_similarity(Z["hog"])`; tương quan Spearman giữa phần tam giác trên của ma trận HOG và ma trận màu (`scipy.stats.spearmanr`).
- Ra: heatmap HOG đặt cạnh heatmap màu + hệ số Spearman.
- Dẫn tới: tương quan thấp → hai đặc trưng "thấy" các món khác nhau → **có cơ sở để gộp**, trước khi huấn luyện mô hình nào.

**22. HOG tìm ra ảnh giống về bố cục hay giống về món?**
- Làm: 4 ảnh train nhóm món sợi; tìm 5 ảnh gần nhất theo HOG, theo màu, theo CLIP (`NearestNeighbors` trên `Z[k][TR]`).
- Ra: 3 lưới (mỗi đặc trưng một lưới) 4 × (1 + 5), tiêu đề ghi tên món, ô sai món tô viền đỏ.
- Dẫn tới: minh họa trực quan: HOG trả về "tô tròn chụp từ trên" của món khác; CLIP trả về đúng món.

## F. CLIP/SigLIP và tính bổ trợ

**23. Đặc trưng nào tách các món tốt nhất?**
- Làm: silhouette và t-SNE đã có sẵn (`eda/tsne_features.png`, `eda_summary.json`). Thêm độ thuần kNN tổng (k = 10, trên val) cho 4 đặc trưng.
- Ra: bảng xếp hạng 4 đặc trưng theo 2 chỉ số.
- Dẫn tới: dự đoán thứ tự kết quả E1/E2 (kỳ vọng CLIP/SigLIP ≫ HOG, màu).

**24. ★ Có món nào HOG hoặc màu làm tốt hơn CLIP không?**
- Làm: độ thuần kNN (k = 10, trên val) theo **món × đặc trưng**.
- Ra: heatmap món × {màu, HOG, CLIP, SigLIP}; đánh dấu ô mà HOG hoặc màu ≥ CLIP.
- Dẫn tới: **bằng chứng sớm nhất cho E7**. Có món mà đặc trưng thủ công thắng CLIP nghĩa là CLIP có điểm mù, và gộp có thể bù vào.

**25. ★ Ở mức từng ảnh, HOG đúng được bao nhiêu ảnh mà CLIP sai?**
- Làm: 1-NN trên val cho mỗi đặc trưng → mảng đúng/sai. Lập bảng 2×2 CLIP × HOG, CLIP × màu, HOG × màu:
  n11 (cả hai đúng), n10, n01, n00 (cả hai sai); oracle = 1 − n00/N; Yule's Q.
  Dùng hàm `contingency`, `yule_q` trong notebook (ô 0.2), chép sang.
- Ra: bảng 3 cặp.
- Dẫn tới: **dự báo trần lợi ích của gộp** trước khi chạy E7. n01 (CLIP sai, HOG đúng) lớn và Q nhỏ → gộp có chỗ để cải thiện.
  Sau này đối chiếu với `results/e7_complementarity.csv` của Người 3.

**26. ★ Cặp món CLIP nhầm nhiều nhất có phải cặp HOG/màu nhầm không?**
- Làm: ma trận nhầm lẫn của 1-NN trên val cho CLIP, HOG, màu; lấy 10 cặp nhầm nhiều nhất của mỗi đặc trưng; đếm cặp chung.
- Ra: bảng 3 cột (top 10 cặp mỗi đặc trưng), tô màu các cặp chung + ma trận nhầm lẫn nhóm món sợi của CLIP.
- Dẫn tới: CLIP nhầm ở chỗ khác HOG/màu thì gộp giúp được. Nếu CLIP nhầm phở ↔ hủ tiếu còn HOG thì không, đó là ví dụ đắt giá.

**27. Ảnh mà mọi đặc trưng đều đoán sai trông thế nào?**
- Làm: ảnh val mà 1-NN của cả 4 đặc trưng đều sai; xem 24 ảnh, phân loại bằng tay: nhãn sai / nhiều món / ảnh ghép / khó thật.
- Ra: lưới ảnh + bảng tỉ lệ 4 loại.
- Dẫn tới: tách **nhiễu nhãn** khỏi **độ khó thật**: phần sai do nhãn sai thì không mô hình nào sửa được.

## G. Ngôn ngữ và câu mô tả

Câu 29–30 cần `vnfood-captions` (Người 1). Câu 28 làm ngay được, nhưng cần GPU để chạy bộ mã hóa văn bản của CLIP.

**28. CLIP/SigLIP biết tên những món Việt nào?**
- Làm: zero-shot trên **val** với 3 cách đặt tên: tiếng Việt có dấu, không dấu, tiếng Anh (dùng `encode_text`, `PROMPTS` trong notebook B3). Tính độ chính xác theo món.
- Ra: biểu đồ cột theo món × 3 cách đặt tên, cho CLIP và SigLIP.
- Dẫn tới: món nào CLIP "không biết tên" (zero-shot gần 0) nhưng 1-NN vẫn tốt → CLIP nhận ra món qua hình nhưng không nối được với tên → dự đoán kết quả E2 zero-shot và E3.

**29. Bộ câu mô tả có chất lượng không?**
- Làm: độ dài (số từ) theo món và theo người viết; 15 từ hay dùng nhất mỗi món; cosine giữa câu và câu mẫu "ảnh món {tên}" của món đúng. Cosine cao bất thường thì có thể câu đang lộ tên món.
- Ra: histogram độ dài + bảng từ khóa theo món + danh sách câu nghi lộ món.
- Dẫn tới: câu lộ món thì báo Người 1 sửa trước khi Người 3 chạy E3.

**30. Câu mô tả có "gần" ảnh đúng món hơn ảnh sai món không?**
- Làm: với câu **val**, cosine với ảnh val đúng món và ảnh val sai món, cho CLIP/SigLIP × có dấu / không dấu / tiếng Anh.
- Ra: 6 cặp histogram (đúng / sai chồng lên nhau) + khoảng cách giữa hai trung vị.
- Dẫn tới: hai phân bố tách nhau càng rõ thì tìm kiếm càng tốt. Dự đoán kết quả E3 và chọn được ngưỡng "không chắc" hợp lý.

---

## Nộp gì

- Mỗi câu: 1 hình PNG (`eda_q01_....png`) hoặc 1 bảng CSV, kèm **1–2 câu kết luận**, ghi chung vào một file `EDA-KET-QUA.md`.
- Đầu file `EDA-KET-QUA.md`: **5 phát hiện quan trọng nhất** và danh sách vấn đề dữ liệu (nếu có). Gửi Người 3, kèm một bản cho Người 1.
