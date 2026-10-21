# ULB Credit Card 数据卡

数据来自 Kaggle 上的 ULB Credit Card Fraud Detection 发布页，许可为 DbCL 1.0。原始 CSV 的大小、SHA-256、清洗统计和划分统计见 `data/manifests/ulb_creditcard.json`。

该数据只有交易相对时间、28 个 PCA 匿名分量、金额和交易级类别，没有客户 ID、卡号、商户 ID或独立调查确认时间。因此只能用于交易级极端不平衡分类，不能构造可信客户序列，也不能用于恢复客户级 `t_f`。

适配器删除完全重复行后按时间顺序划分约 70%/10%/20%，相同时间戳不会跨集合。V1--V28 原样保留；Time 与 `log1p(Amount)` 使用仅从训练集估计的 median/IQR 做稳健缩放。生成文件不是 SafePlus 序列 NPZ，而是包含二维 `x`、交易标签 `y`、原始相对时间 `time` 和清洗后行号 `clean_row_id` 的交易级数据包。
