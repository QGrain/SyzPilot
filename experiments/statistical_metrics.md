# 论文 Statistical Metrics 指标补充

## 实验设置

已知我们将会给我们的方法和各个 baselines 分别记录一份原始实验数据，数据格式如下：

```csv
# Collect the original results of 10 runs and caculate the average results for them.
# The format or original results is as RUN1/RUN2/RUN3/.../RUN10
ID,Hit Count(24h),Avg Hit Count(24h),TTH(24h),μTTH(24h),Hit Count(48h),Avg Hit Count(48h),TTH(48h),μTTH(48h),TTE(24h),μTTE(24h),TTE(48h),μTTE(48h)
example,0/0/0/0/0/0/0/0/0/0,0,...
```

我们除了将上述原始数据整合为论文中已有的 Table II Target Reachability 和 Table III Bug Reproduction 实验结果表格以外，还需要增加以下统计性指标数据表格。

其中：

- SP：SyzPilot
- SN：SnowPlow
- SD：SyzDirect
- SK：Syzkaller
- TTH：Time-to-Hit
- TTE：Time-to-Exposure
- $N_h$：Number of Hits

Benchmark 共包含 70 个 Targets，每个方法在每个 Target 上进行了 10 runs 重复运行。未标记 24h 或 48h 角标的实验默认运行 24h。48h 实验用于补偿部分 baselines 未使用 GPU 所带来的计算资源差异。

论文的主要统计结论统一基于相同 24h budget 的比较；48h 结果作为补充性的 compute-compensated comparison。

Table II 和 Table III 是 case-level detail tables，不强制展示全部 70 个 Targets。Table II 仅展示至少一个被展示 fuzzer/configuration 在至少一个 run 中 hit target PC 的 Targets；所有被展示配置（包括补充性的 48h 配置）的 Hit Count 全为 0 的 Targets 直接省略。Table III 仅展示至少一个被展示 fuzzer/configuration 在至少一个 run 中 exposed target bug 的 Targets；所有被展示配置的 TTE 均为对应 timeout 的 Targets 直接省略。

Statistical summary 同时报告 target-level 和 run-level 成功率。$P_t^h$ 与 $P_t^e$ 的分母均为完整 benchmark 的 70 个 Targets，$P_r^h$ 与 $P_r^e$ 的分母均为实际执行的 $70\times10=700$ runs。所有 fuzzers 均失败的 Targets 仍计入这些成功率的分母。TTH Speedup、$N_h$ improvement、对应的 95% CI、$A_{12}$ 和显著性检验则继续使用仅由 24h 配置确定的有效 target sets $E_{\mathrm{II},24h}$ 与 $E_{\mathrm{III},24h}$。这两个 statistical sets 与 case-level tables 的展示集合分别记录，因而在仅有 48h 配置成功时可能不同。

Table II 的 case-level $N_h$ 单元格为了控制表格宽度而显示为整数：0 保持为 0，其余值四舍五入，且小于 0.5 的正数显示为 1。该显示取整不进入任何统计计算；$N_h$ point estimate、置信区间、$A_{12}$ 和 permutation test 均直接使用未取整的原始 run-level Hit Count。

### Target Reachability 实验

| Fuzzer | Hit Targets ($P_t^h$) | Hit Runs ($P_r^h$) | TTH Spd. [95% CI] | $A_{12}^{TTH}$ | $N_h$ Imp. [95% CI] | $A_{12}^{N_h}$ | $p_{Holm}$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SP | ... | ... | ... | ... | ... | ... | ... |
| SN | ... | ... | ... | ... | ... | ... | ... |
| SD | ... | ... | ... | ... | ... | ... | ... |
| SK | ... | ... | ... | ... | ... | ... | ... |

最后一列泛指本轮比较中的 Holm-adjusted p-value。TTH 与 $N_h$ 仍属于不同的 hypothesis families，不能在统计意义上合并。当前二者数值相同时，表中仅显示一个值以节省宽度；若后续结果不同，生成脚本会在同一行按 TTH/$N_h$ 顺序同时显示两个值并发出提示。

### Bug Reproduction 实验

| Fuzzer | Exposed Targets ($P_t^e$) | Exposed Runs ($P_r^e$) | TTE Spd. [95% CI] | $A_{12}^{TTE}$ | $p_{Holm}$ |
|---|---:|---:|---:|---:|---:|
| SP | ... | ... | ... | ... | ... |
| SN | ... | ... | ... | ... | ... |
| $SD_{24h}$ | ... | ... | ... | ... | ... |
| $SK_{24h}$ | ... | ... | ... | ... | ... |

四个成功率定义为：

$$
P_t^h=\frac{\text{Hit Targets}}{70},
\qquad
P_r^h=\frac{\text{Hit Runs}}{700},
\qquad
P_t^e=\frac{\text{Exposed Targets}}{70},
\qquad
P_r^e=\frac{\text{Exposed Runs}}{700}.
$$

## 各统计性指标的计算方法

### log1p-scale geometric relative improvement for $N_h$

$N_h$ 为非负计数指标，数值越大越好，并且可能为 0。由于不同 Targets 的 $N_h$ 数值尺度可能存在较大差异，因此以 Target 为等权统计单位，而不是直接将所有 Hit Count 求和后比较。

对于 Target $t$，首先计算 SP 和 baseline $F$ 的 10-run 平均值：

$$
\bar{N}_{SP,t}
=
\frac{1}{10}
\sum_{r=1}^{10} N_{SP,t,r},
\qquad
\bar{N}_{F,t}
=
\frac{1}{10}
\sum_{r=1}^{10} N_{F,t,r}.
$$

计算 Target-level log1p effect：

$$
d_t^{N_h}
=
\log(1+\bar{N}_{SP,t})
-
\log(1+\bar{N}_{F,t}).
$$

对于 $N_h$ comparison，定义 Table II 对应的 24h 有效 target set：

$$
E_{\mathrm{II},24h}
=
\left\{
t \mid
\exists m,\exists r,\ N_{m,t,r}^{24h}>0
\right\}.
$$

如果所有被统计的 24h fuzzers 在某个 Target 上所有 runs 的 Hit Count 都为 0，则该 Target 不进入 $E_{\mathrm{II},24h}$，也不参与 $N_h$ improvement、$N_h$ 95% CI、$A_{12}^{N_h}$ 和 $N_h$ permutation p-value 的计算。若某个补充性的 48h 配置成功，该 Target 仍可出现在 case-level Table II 中。

随后对 $E_{\mathrm{II},24h}$ 中的 Targets 等权平均，并转换为 improvement factor：

$$
G^{N_h}
=
\exp
\left(
\frac{1}{|E_{\mathrm{II},24h}|}
\sum_{t \in E_{\mathrm{II},24h}} d_t^{N_h}
\right).
$$

结果以 x-times 形式报告，而不是百分比。例如 $G^{N_h}=1.25$ 报告为 `1.25x`；$G^{N_h}>1$ 表示 SP 的 $N_h$ 更优，$G^{N_h}<1$ 表示 baseline 更优。

表格中的结果例如：

`2.37x [1.82x, 3.06x]`

该指标实际表示 log1p-scale geometric improvement factor。

### 各个 95% CI（包括 TTH、$N_h$ 和 TTE）

所有 95% CI 均采用 Target/run hierarchical bootstrap。Target 是主要统计单位，10 runs 是同一 Target 内的重复实验，因此不能将 $70\times10=700$ 个 runs 直接视为 700 个独立样本。

建议执行：

$$
B_{\mathrm{boot}} = 10{,}000
$$

次 bootstrap，并使用固定随机种子保证结果可复现。

对于 $N_h$，每轮 bootstrap 从对应的 $E_{\mathrm{II},24h}$ 中有放回抽取相同数量的 Targets；对于每个抽中的 Target，再从其 10 runs 中有放回抽取 10 runs；使用重采样后的原始 run 数据重新计算对应统计量，并按照与原始 point estimate 完全相同的公式聚合 eligible Targets。

对于 TTH/TTE，bootstrap 只在对应的 table effective target set 上执行。每轮 bootstrap 从 eligible target set 中有放回抽取相同数量的 Targets；对于每个抽中的 Target，再从其 10 runs 中有放回抽取 10 runs；随后重新计算 $\mu TTH_{m,t}^{*}$ 或 $\mu TTE_{m,t}^{*}$，再计算 target-level ratio，并最终取这些 ratios 的几何均值。不能先对全部 Targets 求一个全局算术均值，再用两个全局均值相除来生成 speedup 或其 95% CI。

最终取 bootstrap distribution 的 2.5% 和 97.5% 分位数：

$$
CI_{95\%}
=
\left[
Q_{0.025},
Q_{0.975}
\right].
$$

对于 $N_h$，每轮 bootstrap 重新计算：

$$
G_b^{N_h,*}
=
\exp
\left[
\frac{1}{|E_{\mathrm{II},24h}|}
\sum_{t \in E_{\mathrm{II},24h}}
\left(
\log(1+\bar{N}_{SP,t}^{*})
-
\log(1+\bar{N}_{F,t}^{*})
\right)
\right].
$$

对于 TTH，24h 主实验中未成功 Hit 的 run 统一按 24h timeout 处理，不能删除失败 run：

$$
\widetilde{TTH}_{m,t,r}
=
\begin{cases}
TTH_{m,t,r}, & \text{成功 Hit},\\
24h, & \text{未成功 Hit}.
\end{cases}
$$

每个 Target 的 mean TTH 为：

$$
\mu TTH_{m,t}
=
\frac{1}{10}
\sum_{r=1}^{10}
\widetilde{TTH}_{m,t,r}.
$$

分析脚本还计算 Table II 对应 24h 有效 target set $E_{\mathrm{II},24h}$ 上的描述性 mean $\mu TTH$，供内部核查使用：

$$
\overline{\mu TTH}_{m}
=
\frac{1}{|E_{\mathrm{II},24h}|}
\sum_{t \in E_{\mathrm{II},24h}}
\mu TTH_{m,t}.
$$

该数值是算术均值形式的描述性统计，不进入精简后的统计表，也不得用 $\overline{\mu TTH}_{F} / \overline{\mu TTH}_{SP}$ 计算论文中的 TTH Speedup。

对于 TTH comparison，使用 Table II 对应的 24h 有效 target set：

$$
E_{\mathrm{II},24h}
=
\left\{
t \mid
\exists m,\exists r,\ \widetilde{TTH}_{m,t,r}^{24h}<24h
\right\}.
$$

如果所有被统计的 24h fuzzers 在某个 Target 上的所有 TTH runs 都是 timeout，则该 Target 不进入 $E_{\mathrm{II},24h}$，也不参与 TTH Speedup、TTH 95% CI、$A_{12}^{TTH}$ 和 TTH permutation p-value 的计算。若某个补充性的 48h 配置成功，该 Target 仍可出现在 case-level Table II 中。

SP 相对于 baseline $F$ 的整体 TTH Speedup 定义为 eligible targets 上 target-level mean-time ratio 的几何均值：

$$
S^{TTH}
=
\exp
\left[
\frac{1}{|E_{\mathrm{II},24h}|}
\sum_{t \in E_{\mathrm{II},24h}}
\log
\left(
\frac{\mu TTH_{F,t}}
{\mu TTH_{SP,t}}
\right)
\right].
$$

注意该定义采用 $\mu TTH_{F,t} / \mu TTH_{SP,t}$ 的方向，因此 $S^{TTH}>1$ 表示 SP 更快。若反过来使用 $\mu TTH_{SP,t} / \mu TTH_{F,t}$，则得到的是 slowdown factor，不是本文表格中的 speedup。

TTE 与 TTH 采用相同方法，只需将 Hit 替换为 Exposure；未成功 Exposure 的 run 统一按 24h timeout 处理，记为 $\widetilde{TTE}_{m,t,r}=24h$。

$$
\overline{\mu TTE}_{m}
=
\frac{1}{|E_{\mathrm{III},24h}|}
\sum_{t \in E_{\mathrm{III},24h}}
\mu TTE_{m,t}.
$$

同样，$\overline{\mu TTE}_{m}$ 只作为内部描述性均值，不进入精简后的统计表；TTE Speedup 仍按 target-level mean-time ratio 的几何均值计算。对于 TTE comparison，使用 Table III 对应的 24h 有效 target set：

$$
E_{\mathrm{III},24h}
=
\left\{
t \mid
\exists m,\exists r,\ \widetilde{TTE}_{m,t,r}^{24h}<24h
\right\}.
$$

如果所有被统计的 24h fuzzers 在某个 Target 上的所有 TTE runs 都是 timeout，则该 Target 不进入 $E_{\mathrm{III},24h}$，也不参与 TTE Speedup、TTE 95% CI、$A_{12}^{TTE}$ 和 TTE permutation p-value 的计算。若某个补充性的 48h 配置成功，该 Target 仍可出现在 case-level Table III 中。

$$
S^{TTE}
=
\exp
\left[
\frac{1}{|E_{\mathrm{III},24h}|}
\sum_{t \in E_{\mathrm{III},24h}}
\log
\left(
\frac{\mu TTE_{F,t}}
{\mu TTE_{SP,t}}
\right)
\right].
$$

$N_h$、TTH 和 TTE 的 95% CI 均从对应的 hierarchical bootstrap distribution 中获得。

论文主要 inference 使用 24h comparison。$SD_{48h}$ 和 $SK_{48h}$ 作为补充实验，不应直接采用 “SP failure = 24h、baseline failure = 48h” 的方式计算主结论中的 TTH/TTE Speedup，否则双方均失败时会人为产生 $48/24=2\times$ 的虚假 speedup。

### $A_{12}$

采用 Vargha-Delaney $A_{12}$ 作为 effect size。

由于不同 Targets 的难度和数值范围明显不同，不将全部 700 个 runs 合并计算一个全局 $A_{12}$。应先在每个 Target 内计算 $A_{12,t}$，再对 Targets 等权平均。

对于 $N_h$，使用 Table II 对应的 24h 有效 target set：

$$
A_{12}^{N_h}
=
\frac{1}{|E_{\mathrm{II},24h}|}
\sum_{t \in E_{\mathrm{II},24h}} A_{12,t}^{N_h}.
$$

其中 $N_h$ 数值越大越好：

$$
A_{12,t}^{N_h}
=
\frac{
\sum_{i=1}^{10}
\sum_{j=1}^{10}
\left[
I(x_i>y_j)
+
0.5I(x_i=y_j)
\right]
}
{100},
$$

其中 $x_i$ 为 SP 的 run，$y_j$ 为 baseline 的 run。

对于 TTH 和 TTE，数值越小越好，因此比较方向反转，并且仅在对应的 table effective target set 上等权平均：

$$
A_{12}^{TTH}
=
\frac{1}{|E_{\mathrm{II},24h}|}
\sum_{t \in E_{\mathrm{II},24h}} A_{12,t}^{TTH},
\qquad
A_{12}^{TTE}
=
\frac{1}{|E_{\mathrm{III},24h}|}
\sum_{t \in E_{\mathrm{III},24h}} A_{12,t}^{TTE}.
$$

$$
A_{12,t}^{TTH/TTE}
=
\frac{
\sum_{i=1}^{10}
\sum_{j=1}^{10}
\left[
I(x_i<y_j)
+
0.5I(x_i=y_j)
\right]
}
{100}.
$$

TTH/TTE 使用与 Speedup 相同的 timeout-penalized run values。

所有指标统一保证：

$A_{12}>0.5$ 表示 SP 更优；

$A_{12}=0.5$ 表示双方无明显优势；

$A_{12}<0.5$ 表示 baseline 更优。


### Holm-adjusted stratified permutation p-values

所有显著性检验首先采用 **Target-stratified permutation test** 计算 raw p-value，再使用 **Holm correction** 对同一指标下的多个 baseline comparisons 进行多重检验校正。论文表格最终报告 Holm-adjusted stratified permutation p-values。

对于每个 baseline $F$，以 Target 作为 stratum。每个 Target 内包含 SP 和 $F$ 各 10 个 runs。Permutation 仅在同一 Target 内进行：将该 Target 的 20 个 run values 合并后随机置换方法标签，并重新划分为 10 个 SP runs 和 10 个 baseline runs。不同 Targets 之间不得交换样本，从而保留不同 Targets 固有的难度和数值尺度差异。对于 $N_h$、TTH 和 TTE，permutation test 均仅在对应的 table effective target set 上执行。

建议执行

$$
B_{\mathrm{perm}} = 100{,}000
$$

次 permutation，并使用固定随机种子保证结果可复现。

检验统计量与对应 point estimate 的聚合方式保持一致，并统一在 log-effect scale 上进行检验。

对于 $N_h$，observed statistic 定义为：

$$
T_{\mathrm{obs}}^{N_h}
=
\frac{1}{|E_{\mathrm{II},24h}|}
\sum_{t \in E_{\mathrm{II},24h}}
\left[
\log(1+\bar{N}_{SP,t})
-
\log(1+\bar{N}_{F,t})
\right].
$$

对于 TTH，observed statistic 定义为：

$$
T_{\mathrm{obs}}^{TTH}
=
\frac{1}{|E_{\mathrm{II},24h}|}
\sum_{t \in E_{\mathrm{II},24h}}
\log
\left(
\frac{\mu TTH_{F,t}}
{\mu TTH_{SP,t}}
\right).
$$

对于 TTE：

$$
T_{\mathrm{obs}}^{TTE}
=
\frac{1}{|E_{\mathrm{III},24h}|}
\sum_{t \in E_{\mathrm{III},24h}}
\log
\left(
\frac{\mu TTE_{F,t}}
{\mu TTE_{SP,t}}
\right).
$$

TTH 和 TTE 使用与 Speedup 及 $A_{12}$ 相同的 timeout-penalized run values 与 table effective target set。对于 24h 主实验，eligible target 中未成功 Hit 或 Exposure 的 run 仍统一赋值为 24h timeout；只有所有被统计的 24h fuzzers 在该 Target 上全部 timeout 时，才将该 Target 从对应的 time-to-event statistics 中排除。

在每一次 permutation $b$ 中，对每个 Target 独立置换 SP 和 baseline 的方法标签，然后使用置换后的 run values 重新计算完整统计量，得到

$$
T_b^*.
$$

零假设为：在控制 Target 分层结构后，SP 与 baseline $F$ 在该指标上不存在系统性差异，即

$$
H_0: T = 0.
$$

采用双侧 permutation test。对于每个 SP-baseline comparison，raw p-value 计算为：

$$
p_{\mathrm{raw}}
=
\frac{
1
+
\sum_{b=1}^{B_{\mathrm{perm}}}
I\left(
|T_b^*|
\ge
|T_{\mathrm{obs}}|
\right)
}{
B_{\mathrm{perm}}+1
}.
$$

其中 $I(\cdot)$ 为 indicator function。分子和分母中的加一用于避免 Monte-Carlo permutation 得到 $p=0$。

在 $B_{\mathrm{perm}}=100{,}000$ 时，raw permutation p-value 的最小可报告值为

$$
\frac{1}{100{,}000+1}
\approx
10^{-5}.
$$

如果同一 hypothesis family 中有 3 个 baseline comparisons，则最小 Holm-adjusted p-value 可能达到

$$
\frac{3}{100{,}000+1}
\approx
3 \times 10^{-5}.
$$

因此，报告中出现约 $3\times10^{-5}$ 的 Holm-adjusted p-value 并不必然异常，它通常表示在 100,000 次 permutation 中没有任何一次置换统计量达到或超过 observed statistic，并且 Holm correction 将该 raw p-value 乘以了当前 family size。若需要更细的 p-value 分辨率，可以增加 permutation 次数；否则也可以在论文中报告为 $p_{\mathrm{Holm}} < 3\times10^{-5}$。

随后，对同一 outcome 下多个 baseline comparisons 得到的 raw p-values 使用 Holm procedure 进行多重检验校正。

对于 24h 主实验，TTH、$N_h$ 和 TTE 分别构成独立的 hypothesis family。例如：

$$
\mathcal{P}^{TTH}
=
\left\{
p_{SN}^{TTH},
p_{SD_{24h}}^{TTH},
p_{SK_{24h}}^{TTH}
\right\},
$$

$$
\mathcal{P}^{N_h}
=
\left\{
p_{SN}^{N_h},
p_{SD_{24h}}^{N_h},
p_{SK_{24h}}^{N_h}
\right\},
$$

以及

$$
\mathcal{P}^{TTE}
=
\left\{
p_{SN}^{TTE},
p_{SD_{24h}}^{TTE},
p_{SK_{24h}}^{TTE}
\right\}.
$$

因此，每个 24h hypothesis family 中共有

$$
m = 3
$$

个 comparisons。

设同一 family 中的 raw p-values 按从小到大排序为：

$$
p_{(1)}
\le
p_{(2)}
\le
\cdots
\le
p_{(m)}.
$$

对应的 Holm-adjusted p-values 为：

$$
p_{(i)}^{\mathrm{Holm}}
=
\min
\left(
1,
\max_{1 \le j \le i}
\left[
(m-j+1)p_{(j)}
\right]
\right),
\qquad
i=1,\ldots,m.
$$

计算完成后，将排序后的 adjusted p-values 映射回其原始的 SP-baseline comparisons。

因此，论文表格中的 $p\text{-value}^{TTH}$、$p\text{-value}^{N_h}$ 和 $p\text{-value}^{TTE}$ 均报告对应的 **Holm-adjusted stratified permutation p-value**，而不是未经校正的 raw permutation p-value。

SD$_{48h}$ 和 SK$_{48h}$ 属于补充性的 compute-compensated comparison，不与 24h 主实验中的 comparisons 共同组成 Holm family。若对 48h comparison 报告显著性检验，则对应的 raw permutation p-values 应单独构成 supplementary hypothesis family，并独立进行 Holm correction。

需要特别注意，TTH/TTE 的 permutation test 必须建立在双方可比较的 timeout 编码规则上。不得简单使用“SP failure = 24h、baseline failure = 48h”的不对称 timeout 编码进行显著性检验，否则双方均失败时也会人为产生 baseline/SP = 2 的比值，从而引入虚假的 speedup effect。

论文统一采用

$$
\alpha = 0.05
$$

作为显著性水平。当

$$
p_{\mathrm{Holm}} < 0.05
$$

时，认为在控制 Target 分层结构以及同一指标下多个 baseline comparisons 所产生的 multiple-testing effect 后，SP 与对应 baseline 之间仍存在统计显著差异。

p-value 仅用于判断统计显著性，不用于衡量实际差异大小。实际 effect magnitude 分别由 TTH/TTE Speedup、$N_h$ improvement factor、95% CI 和 $A_{12}$ 共同描述。
