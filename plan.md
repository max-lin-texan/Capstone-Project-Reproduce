# Capstone Project Analysis Plan

## 研究主軸

依照教授回信中對病人異質性和 hospital comparison 的方向，本研究要回答：

- 是否存在不同的 sepsis 病人群組？
- 不同群組的生理變數和時間變化有何不同？
- 不同醫院是否出現相似的病人群組？
- 哪些模式是所有醫院共有，哪些只出現在特定醫院？
- CPD 與 NNCPD、以及 mean/median/last 是否得到穩定且相近的結論？

重要區分：

- `rank` 是 CPD/NNCPD 用來表示資料的 latent component 數量。
- patient clustering 是使用 `rankX_patient.csv` 的 component scores，把病人分成生理模式相似的群組。
- rank 不等於病人群組數量。

## 教授回信確認的原則

- 不使用把所有 rank 或所有模型混在一起的 pooled cosine similarity 結果。
- 先用 fit、relative error、rank complexity、random-start stability 等條件選出少數較好的模型，再對這些模型做比較和繪圖。
- 目前使用 fit 或 relative error 時，不需要另外用 RMSE 計算 fit；RMSE 可以保留作補充診斷，但不是模型選擇的必要依據。
- 暫時不做 quantile CPD。先完成目前的 CPD/NNCPD、模型選擇、combined images 和 patient clustering，之後有時間再考慮 quantile CPD。
- 教授所說的 large/combined map，是將 soft-clustering 圖和 CPD component 圖用一致版面排列，讓 hospital 與 overall 可以直接比較，不是把 cosine similarity 當成唯一的圖。
- patient mode 是原本主要感興趣的分析方向。要研究不同病人是否以不同方式經歷 sepsis，且要同時分析 overall model 和各 hospital model。
- patient mode 不必限制使用 cosine similarity；可以比較 K-means、hierarchical clustering 等方法，再比較分群結果。

## 已完成

### 1. 資料與 cohort 建立

- [x] 完成 feature 7 與 feature 11 的 preprocessing/imputation。
- [x] 完成 mean、median、last 三種 aggregation。
- [x] 建立 `overall` 與 `hospital_1` 到 `hospital_10` 的 cohort。
- [x] 建立 [hospital_map.csv](results/hospital_map.csv) 保存 hospital 對照關係。

### 2. Rank sweep 與 factor 分解

- [x] CPD 測試 rank 1--15。
- [x] NNCPD 測試 rank 1--15。
- [x] 每個模型使用多個 random starts，記錄最佳 fit 和 fit variability。
- [x] 儲存 patient、variable、time factor 以及 component weights。
- [x] 儲存 `fit_summary.csv` 和 `fit_vs_rank.png`。

### 3. Cohort similarity comparison

- [x] 比較 overall 和各 hospital 的 feature loading similarity。
- [x] 比較 overall 和各 hospital 的 time loading similarity。
- [x] 產生 cosine similarity CSV 與 heatmap。
- [x] CPD/NNCPD、feature set、aggregation 和 rank 分開比較，不將不同設定混在一起。

限制與後續修正：

- [x] 已停止 pooled cosine similarity 的方向；目前 cosine similarity 只在相同 feature set、method、aggregation 和 rank 的 cohort 之間比較。
- [ ] 後續只對第二層選出的少數主要模型保留正式 cosine comparison 和 heatmap。

### 4. 第一層 rank selection

- [x] 建立 [model_selection.csv](results/model_selection.csv)，共記錄 1,980 個候選模型。
- [x] 建立 [model_diagnostics.csv](results/model_diagnostics.csv)，摘要每個設定的 selected rank、elbow 和穩定性。
- [x] 每個分析設定選出一個 selected rank，共 132 個 selected models。
- [x] 目前的自動規則是：在 fit 接近最佳、random starts 穩定的 rank 中，選最低的 rank。
- [x] 執行 cleanup，刪除未選 rank 的 factor 檔案，保留 selected rank 的輸出。

目前結果的解讀：

- CPD 多數 selected rank 約為 14--15，表示在 rank 15 上限內仍可能持續改善。
- NNCPD 多數 selected rank 約為 9--12，通常較早達到穩定結果。
- CPD 與 NNCPD 的 fit 不可直接互相比較，因為兩者使用不同 scaling 和 constraint。

## 尚未完成與下一步

### 5. 第二層：從 132 個 selected models 選主要研究模型

這一步與「每個設定選一個 rank」不同。現在已完成每個設定內的 rank selection，但尚未決定最後要用哪些模型作為主要研究結果。

- [x] 在 [model_selection.py](src/capstone_project_reproduce/model_selection.py) 實作第二層模型選擇，輸出 [model_selection_summary.csv](results/model_selection_summary.csv)。
- [x] 使用 component-aligned cosine similarity 比較 primary 與 sensitivity model 的 feature/time loading 穩定性。
- [x] 定義主要分析的模型組合，而不是直接對全部 132 個模型做 clustering。
- [x] 先選一組代表模型：
	- [x] overall cohort 作為主要分析。
	- [x] CPD 與 NNCPD 各保留一組比較。
	- [x] 先選一個主要 feature set，另一個作 sensitivity analysis。
	- [x] mean/median/last 中選主要 aggregation，其餘作 stability comparison。
- [ ] 讀取 selected model 的 factor 圖與 `rankX_variable.csv`、`rankX_time.csv`。
- [ ] 依照下列條件人工確認主要模型：
	- fit 足夠高且 relative error 可接受。
	- rank 不過度複雜。
	- random starts 穩定。
	- components 的 feature/time loading 有清楚且可解釋的 pattern。
	- 結果在不同 aggregation 或 nearby model 間不會完全改變。
- [ ] 必要時為主要模型保留一個備選 rank，並記錄選擇理由。
- [ ] 建立一個明確的 final model table，記錄 primary model、sensitivity model 和排除理由。

選擇結果應分成兩類：

- **Primary models**：用於主要結論、combined images 和 patient clustering。
- **Sensitivity models**：用於確認 CPD/NNCPD、feature set、aggregation 或 nearby model 改變時，結論是否穩定。

不要把 132 個 selected models 全部當成同等重要的主要模型，也不要在尚未選出代表模型前產生大量正式圖片。

### 6. Selected-model visualization

只針對第二層選出的主要模型製作正式圖表：

- [x] 新增 [selected_model_visualization.py](src/capstone_project_reproduce/selected_model_visualization.py)，從 [model_selection_summary.csv](results/model_selection_summary.csv) 動態讀取 `second_level_selected=True` 的 primary models，不寫死 method、aggregation 或 rank。
- [x] feature loading plots。
- [x] time loading plots。
- [x] patient factor distribution plots。
- [x] overall 與 hospital 的 aligned combined feature maps。
- [x] 固定 component 對齊方式、顏色範圍、圖片尺寸、標題和圖例。
- [x] 輸出至 `results/selected_model_visualizations/`；結果資料夾分為 `results/cosine_similarity/`、`results/cpd&nncpd/`、`results/model_selection/` 和 `results/selected_model_visualizations/`。

Combined image 的設計：

- [ ] 以 variables 或 components 作為一個軸。
- [ ] 以 `hospital_1` 到 `hospital_10` 加上 `overall` 作為另一個軸。
- [ ] 每個 cell 顯示同一變數/component 在該 hospital 或 overall 的結果。
- [ ] soft clustering 與 CPD component images 使用一致的排列、尺度、顏色和標題。
- [ ] 這張圖的目的，是讓人同時比較各 hospital 和 overall 的 pattern；它不等同於 cosine similarity heatmap。

圖表應清楚區分：現有的 ICD/organ-based `clustering/`，以及後續根據 patient factors 做的 patient clustering。

### 7. Patient-factor clustering

這是教授研究方向下的重要後續分析，應使用第二層選出的主要模型，不要盲目分析全部 132 個模型。

 [x] 新增 [patient_clustering.py](src/capstone_project_reproduce/patient_clustering.py)，動態讀取 `model_selection_summary.csv` 的 primary models。
 [x] 讀取 selected model 的 `rankX_patient.csv`。
 [x] 保留 `patientunitstayid`，並為每位病人記錄 cluster label。
 [x] 對 component scores 做 standardization。
 [x] 比較 K-means 與 hierarchical clustering。
 [x] 測試 `k = 2` 到 `k = 6`。
 [x] 使用 silhouette、Calinski-Harabasz 和 Davies-Bouldin 指標。
 [x] 檢查各 cluster 的人數，標記過小或不穩定的群組。
 [x] 產生 clustering metrics、cluster profiles、cluster size 和 PCA 圖表。
 [x] 建立 [email_report.md](results/patient_clustering/email_report.md) 整理初步結果。
- [x] 新增 exploratory candidate outputs：CPD `k=3` 與 NNCPD `k=4`，並使用高對比 PCA cluster colors。
- [x] 使用 [cluster_stability.py](src/capstone_project_reproduce/cluster_stability.py) 在 overall NNCPD rank9 以 seed 42、43、44、45、46 檢查 k=2、k=4 的 stability。
  - k=2：pairwise ARI 最低約 0.9988，AMI 最低約 0.9956，約 10,501/10,502 位病人達成完整共識。
  - k=4：pairwise ARI 最低約 0.9947，AMI 最低約 0.9833，約 10,486/10,502 位病人達成完整共識。
  - 結果與 consensus labels 已輸出至 [patient_clustering_stability](results/patient_clustering_stability/)。
- [ ] 以 nearby ranks 和 alternative methods 檢查 clustering stability。
- [x] 先完成 overall model 上的分析及 seed stability。
- [ ] 將相同 NNCPD feature_11/last/rank9 設定對各 hospital model 重複。
- [ ] 跨 hospital 比較時處理 cluster label alignment，不能直接假設不同資料中的 cluster 1 代表同一群。
- [ ] 新增 hospital cluster comparison script，對齊 cluster profile 並製作 cross-hospital heatmap。
- [ ] 不把 patient mode 限制成 cosine similarity；以 cluster metrics、cluster profiles 和 stability comparison 判斷結果。

建議輸出：

```text
results/patient_clustering/
	<feature_set>/<cohort>/<method>_<aggregation>/
		cluster_labels.csv
		cluster_metrics.csv
		cluster_profiles.csv
		embedding.png
		cluster_profile_heatmap.png
		cluster_sizes.png
```

### 8. Patient cluster interpretation

對每個主要 cluster 說明：

- [ ] 病人數量和比例。
- [ ] 哪些 CPD components 分數較高。
- [ ] 哪些 clinical variables 的 loading 較重要。
- [ ] 生理變化在時間軸上的差異。
- [ ] 是否存在可重現的 clinical pattern。
- [ ] 必要時比較 outcome 或其他 clinical characteristics，但不能只因 cluster 名稱就做醫療推論。

優先考慮的視覺化：

- PCA 或 UMAP patient-factor embedding，以 cluster label 上色。
- cluster-by-component profile heatmap。
- cluster size bar chart。
- 各 cluster 的 feature/time profile small multiples。
- K-means、hierarchical 和不同 rank 間的 ARI/AMI stability heatmap。
- 若 cluster labels 能合理對齊，再考慮跨 hospital 的 alluvial/Sankey；否則使用 profile similarity matrix。

### 9. Final comparison and conclusions

- [ ] 比較 overall 和各 hospital 的 cluster proportion。
- [ ] 比較 hospital model 和 overall model 的 cluster/profile similarity。
- [ ] 比較 CPD 與 NNCPD 是否得到相近的病人群組。
- [ ] 比較 mean、median、last 是否得到穩定 pattern。
- [ ] 判斷哪些生理模式跨 hospital 共通，哪些是 hospital-specific。
- [ ] 整理研究限制，包括 rank 上限、cluster label alignment、model selection uncertainty 和未驗證的 clinical interpretation。
- [ ] 形成最後的研究結論與報告圖表。

## 執行順序總覽

```text
[已完成] preprocessing / cohort 建立
		-> [已完成] CPD 與 NNCPD rank 1--15
		-> [已完成] factor、fit、cosine similarity 輸出
		-> [已完成] 每個設定選一個 rank：132 個 selected models
		-> [已完成] 第二層 primary/sensitivity model selection
		-> [已完成] selected-model comparison figures
		-> [已完成] overall patient-factor clustering
		-> [已完成] overall seed stability：k=2、k=4 均高度穩定
		-> [下一步] hospital patient-factor clustering 與比較
		-> [下一步] cluster interpretation 和 cross-hospital alignment
		-> [最後] 最終研究結論與報告圖表
```

目前最重要的原則是：先完成清楚且可解釋的第二層 model selection，再進行 patient clustering；不要把 rank、latent component 和 patient cluster 混為同一件事。