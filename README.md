# 量子適用性健檢（Quantum Fit Check）

企業放一小份 CSV，系統在同一份資料上讓量子方法和強的古典方法正面比，用統計檢定判斷輸贏，給紅／黃／綠燈；古典贏的時候，推薦該用哪個古典方法和現成工具。

- **網頁版**：https://johnny-0120.github.io/quantum-fit-check/ （`index.html` 單一檔案，全部在瀏覽器內計算，資料不上傳；下載後直接用瀏覽器打開也可以）。
- **Colab／Python 版**：`colab/qfc_colab.py`（腳本）與 `colab/quantum_fit_check_colab.ipynb`（筆記本），可跑更大的資料。

## 能檢查的問題

| 問題 | 量子方法（無雜訊精確模擬） | 古典方法 |
|---|---|---|
| 最佳化 | QAOA p=1/p=2、CVaR-QAOA、Warm-start QAOA、RQAOA、量子退火、VQE、Grover 適應性搜尋 | 精確窮舉、貪婪／多起點區域搜尋、模擬退火、禁忌搜尋、GW 半定規劃 |
| 分類 | ZZ 量子核函數、投影量子核函數、變分量子分類器 VQC | 邏輯回歸、線性核、RBF 核、k 近鄰、隨機森林、神經網路 |
| 分布學習 | IQP 量子生成模型、硬體高效 QCBM | 各欄獨立、Chow-Liu 樹、伯努利混合、平滑直方圖 |

最佳化可讀三種格式：邊清單（分兩組）、QUBO 方陣、0/1 問題規格表（`kind,a,b,value,penalty`，可含「剛好選 k 個」限制）。範例在 `samples/`。

## 怎麼判斷輸贏

- 最佳化：隨機方法各跑 3 次取最好；量子參數最佳化是確定性的，重複的是 1024 次量測。
- 分類：5 折交叉驗證重複 3 次，最好的量子 vs 最好的古典做 Nadeau–Bengio 修正 t 檢定；綠燈要求 p < 0.05 ÷ 量子模型數（Bonferroni）且差距 ≥ 1 個百分點。
- 分布學習：3 次不同切分，逐筆配對比較測試資料的對數概似；綠燈同樣用 Bonferroni 門檻且差距 ≥ 0.02 bits。
- 所有調參數（核函數頻寬、正則化、VQC 編碼頻寬、平滑係數）都只用訓練資料。

## 校正實驗（`node tests/calibrate.js`）

| 問題 | 資料 | 份數 | 綠燈 |
|---|---|---|---|
| 分類 | 古典規則產生（對照組） | 8 | 0（沒有誤判） |
| 分類 | 標籤由 ZZ 量子核函數產生（植入組） | 4 | 4（全部抓到） |
| 分布學習 | 古典規則產生（對照組） | 4 | 0 |
| 分布學習 | 樣本由 IQP 線路產生（植入組） | 4 | 0（量子模型訓練卡在局部最小值） |

樣本數少，只代表大致趨勢。

## 已知限制

- 量子部分全部是古典電腦上的無雜訊模擬；綠燈代表值得再測，不代表量子優勢。
- 完整量子模擬上限 14 個變數（網頁）／20 個（Colab），這個規模古典窮舉一定解得出，所以小規模最佳化幾乎都是紅燈。這是篩檢工具，主要功能是排除不需要量子的情況。
- 分布學習的量子模型目前訓練不起來，偵測力為 0。
- 資料體檢的門檻（關係強弱差距 10 倍、幾何差距比值 0.3 等）是經驗規則，只用來解釋結果，不決定燈號。
- 所有範例都是合成資料，尚未用真實企業資料驗證。

## 開發

```bash
node tests/verify.js      # 公式、梯度、QUBO 轉換、t 分布檢查
node tests/calibrate.js   # 校正實驗（約 3 分鐘）
node build.js             # template.html + engine.js → index.html
python colab/build_notebook.py   # qfc_colab.py → .ipynb
python colab/qfc_colab.py samples/cloudflare.csv cls   # 需要 numpy scipy pandas scikit-learn（qiskit 選用）
```

## 授權

MIT License，見 `LICENSE`。範例資料皆為程式合成。
