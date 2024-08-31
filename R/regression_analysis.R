# 必要なパッケージをインストール
if (!require("readxl")) install.packages("readxl")
if (!require("survival")) install.packages("survival")
if (!require("dplyr")) install.packages("dplyr")

# パッケージを読み込む
library(readxl)
library(survival)
library(dplyr)

# データを読み込む
file_path <- "/Users/junyasato/Downloads/util_example/41467_2024_47512_MOESM6_ESM.xlsx"
data <- read_excel(file_path, skip = 1) # 1行目をスキップしてヘッダーを正しく読み込む

# 列名の整理
colnames(data) <- c('Patient_id', 'OS_Status', 'OS_Time', 'RFS_Status', 'RFS_Time', 'Tumor_vol_ml', 'label', 'new_Label_meaning')

# 単変量解析
# 数値変数の概要統計量を表示
summary(data$OS_Time)
summary(data$RFS_Time)
summary(data$Tumor_vol_ml)

# カテゴリカル変数のクロス集計とカイ二乗検定
table_os_status <- table(data$OS_Status)
table_rfs_status <- table(data$RFS_Status)
chisq.test(table(data$OS_Status, data$new_Label_meaning))

# tumor_vol_mlを標準化
data$Tumor_vol_ml <- scale(data$Tumor_vol_ml)

# 多変量解析: Cox比例ハザードモデル
# OS_TimeとOS_Statusに基づいて解析
cox_model <- coxph(Surv(OS_Time, OS_Status) ~ Tumor_vol_ml , data = data)
print(summary(cox_model))

# RFS_TimeとRFS_Statusに基づいて解析
cox_model_rfs <- coxph(Surv(RFS_Time, RFS_Status) ~ Tumor_vol_ml , data = data)
print(summary(cox_model_rfs))
print('Multivariate end')

# High riskを基準として単変量解析
data$new_Label_meaning <- factor(data$new_Label_meaning, levels = c("High risk", "Intermediate risk", "Low risk"))
# 単変量Cox比例ハザードモデルの実行
cox_model_univariate_OS <- coxph(Surv(OS_Time, OS_Status) ~ new_Label_meaning, data = data)

# RFS_TimeとRFS_Statusに基づいて解析
cox_model_univariate_RFS <- coxph(Surv(RFS_Time, RFS_Status) ~ new_Label_meaning, data = data)


## 多変量Cox比例ハザードモデルの実行、Tumor_vol_mlで調整
cox_model_adjusted <- coxph(Surv(OS_Time, OS_Status) ~ Tumor_vol_ml + new_Label_meaning, data = data)


# 結果をテキストファイルに保存
sink("statistical_results/cox_model_results.txt")  # 出力をファイルにリダイレクト
print(summary(cox_model_univariate_OS))
print(summary(cox_model_univariate_RFS))
print(summary(cox_model_adjusted))
sink() 