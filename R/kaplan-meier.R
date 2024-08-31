# 必要なパッケージをインストール
if (!require("survminer")) install.packages("survminer")
if (!require("survival")) install.packages("survival")
if (!require("gridExtra")) install.packages("gridExtra")

# パッケージを読み込み
library(survival)
library(survminer)
library(gridExtra)

# new_Label_meaningを因子型 (factor) に変換し、基準レベルを "High risk" に設定
data$new_Label_meaning <- factor(data$new_Label_meaning, levels = c("High risk", "Intermediate risk", "Low risk"))

# Kaplan-Meierの生存曲線を計算
km_fit <- survfit(Surv(OS_Time, OS_Status) ~ new_Label_meaning, data = data)
print(km_fit)

# Kaplan-Meier曲線をプロットし、オブジェクトとして保存
km_plot <- ggsurvplot(km_fit, 
                      data = data,
                      pval = TRUE,           # Log-rankテストのp値を表示
                      conf.int = TRUE,       # 信頼区間を表示
                      risk.table = TRUE,     # リスクテーブルを表示
                      ggtheme = theme_minimal(), # プロットのテーマを設定
                      palette = c("#E7B800", "#2E9FDF", "#E74C3C"), # カスタムパレット
                      title = "Kaplan-Meier Curve for new_Label_meaning"
)

# PDF出力を開始
pdf("statistical_results/Kaplan_Meier_Curve.pdf", width = 16, height = 12)

# プロットを描画
print(km_plot$plot)
print(km_plot$table)

# PDF出力を終了
dev.off()