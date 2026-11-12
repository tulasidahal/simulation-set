::: {custom-style="Title"}
[Experiment, Analysis, and Benchmark] Leakage-Safe Point-in-Time Data Preparation and Evaluation for Direction Prediction from Financial Time Series: A Case Study on the Nepal Stock Exchange
:::

::: {custom-style="Author"}
Tulasi Dahal, Kishor Basyal
:::

::: {custom-style="Affil"}
[[Affiliations]] · [[Corresponding e-mail]] · [[ORCID iDs]]
:::

::: {custom-style="Abstract"}
**Abstract**—Accuracy reported for machine-learning stock predictors often fails to survive deployment, and the cause is frequently a data-management defect rather than a modelling one: features, labels or evaluation splits that let future information reach training. We propose a general pipeline for leakage-safe temporal data preparation and evaluation and study it through a case study on the Nepal Stock Exchange (NEPSE), a thinly traded market with daily price limits and frequent bonus and rights issues. In the case study we build a point-in-time feature store (148,187 stock-days, 101 securities, 3,343 trading days, 72 features) with forward-return labelling, within-date rank normalisation and a suite of integrity audits, and we evaluate four base learners (CatBoost, LightGBM, a customised XGBoost and a Transformer) and their combinations under a purged, embargoed walk-forward protocol with five seeds. Four findings emerge. First, random k-fold validation inflates the Matthews correlation coefficient (MCC) of the same model by about 65% relative to walk-forward evaluation, while a shuffled-label control returns chance. Second, a learned logistic stacker underperforms every single model (MCC 0.037 against 0.071 for the best), whereas averaging calibrated probabilities matches the best model on MCC and has the best log loss (1.074) and calibration error (0.014). Third, two of the XGBoost customisations we expected to help, time-decay weighting and categorical symbol and sector identifiers, reduce MCC. Fourth, 85 day-over-day moves beyond the 10% price band persist in the price series treated as adjusted, which corrupts labels. Predictability is real but small (balanced accuracy 0.38 against 0.33 for chance), and we argue that this scale, rather than a headline accuracy, is the defensible result. Code and protocol are released.
:::

::: {custom-style="Keywords"}
*Index Terms*—temporal data leakage, point-in-time feature store, walk-forward validation, data quality, ensemble calibration, forecast combination, frontier markets, reproducibility.
:::

```{=openxml}
<w:p><w:pPr><w:spacing w:before="0" w:after="0"/><w:sectPr><w:type w:val="nextPage"/><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="900" w:bottom="1440" w:left="900" w:header="720" w:footer="540" w:gutter="0"/><w:cols w:num="1" w:space="360"/></w:sectPr></w:pPr></w:p>
```

# I. Introduction

Published results for machine-learning stock prediction are frequently far better than anything practitioners can reproduce. A recurring explanation is that the evaluation, not the model, is flawed: look-ahead in feature construction, labels that overlap the test period, random cross-validation over autocorrelated series and silent errors in corporate-action adjustment all let future information leak into training {{c:kaufman,kapoor,yang}}. These are data-management problems. They concern how temporal data is versioned, aligned to the moment a decision would have been made, validated and partitioned, and they decide whether a reported number means anything. The data-management community has built tools for the adjacent problems of validating training data {{c:breck,schelter}}, managing features {{c:orr}} and surveying data-quality challenges in ML {{c:polyzotis,whang}}, but temporal leakage in financial prediction has received little attention from this angle.

We study these problems in general and use one market as a case study. The Nepal Stock Exchange (NEPSE) is a demanding test: it lists a few hundred securities, applies daily price-limit bands, trades thinly, and its companies frequently distribute bonus and rights shares that change a price series discontinuously. An unadjusted bonus issue looks like a one-day crash, produces a wrong label, and can be learned as if it were a signal. Short histories and strong regime shifts (a boom, a long decline, a recovery) make it easy to select a configuration that works only on the period used to select it.

We therefore present a general, market-independent pipeline for leakage-safe temporal data preparation and evaluation, instantiate it on NEPSE as a case study, and ask four research questions.

- **RQ1.** How much do common evaluation shortcuts inflate performance on this data?
- **RQ2.** Which data-integrity defects survive in a nominally adjusted price series, and can they be detected automatically?
- **RQ3.** Which model-side customisations survive a rigorous protocol?
- **RQ4.** Does a learned meta-learner improve on simple averaging when every component is calibrated and evaluation is strictly chronological?

**Contributions.** (1) A general point-in-time feature store and audit suite, instantiated on the NEPSE case study: forward-return labelling on adjusted prices, within-date rank normalisation, a corporate-action breach audit against the price-limit band, an information-coefficient leakage scan validated by a planted canary, and a truncation-invariance test for feature code (Sections III–IV). (2) A purged, embargoed walk-forward protocol in which both base learners and the combiner are trained only on data that precedes each test block by the label horizon plus an embargo (Section V). (3) An experimental analysis over five seeds with nine XGBoost ablations, four leave-one-model-out combiners, a leakage-protocol experiment, and block-bootstrap, Diebold–Mariano and McNemar tests with false-discovery-rate control (Section VIII). (4) Null and negative findings reported as such: the learned stacker loses to averaging, two customisations hurt and the embargo length has no consistent effect at a one-day horizon. (5) Released artifacts that regenerate every table and figure (Section X).

**Findings in brief (NEPSE case study).** Predictability is small but present and consistent across four test regimes (balanced accuracy 0.38 against 0.33 for chance; MCC about 0.07). The robust results concern data and evaluation rather than models: random splitting overstates MCC by about 65%; 85 price-band breaches survive in an adjusted series; and a learned combiner can make an ensemble worse than its parts.

# II. Related Work

**Positioning.** This work builds on data-management research at VLDB, SIGMOD and ICDE. Automated data-quality verification for ML pipelines {{c:schelter}}, feature stores {{c:orr}} and the data lifecycle of production ML {{c:polyzotis}} define the problem space. Guha et al. at ICDE 2023 show that data-preparation choices change downstream ML outcomes {{c:guha}}, and the TFB benchmark at VLDB 2024 shows that inconsistent evaluation pipelines distort comparisons of time-series forecasters {{c:tfb}}. We extend these lines to a defect class that none of them targets: temporal leakage and corporate-action errors in financial prediction data, and the evaluation protocol that detects them.

::: {custom-style="TabCap"}
TABLE I. Related work and the gap addressed here.
:::

| Line of work | Representative studies | What it provides | Gap addressed here |
|:--|:--|:--|:--|
| Data validation, feature stores | {{c:breck,schelter,orr,polyzotis}} | Schema, drift and quality checks; feature management | No test for look-ahead or label errors in temporal data |
| Leakage | {{c:kaufman,kapoor,yang}} | Definitions, prevalence, static detection | Not applied to financial labels or to combiners |
| Time-ordered evaluation | {{c:bergmeir12,bergmeir18,cerqueira,lopez}} | Valid splits, purging, embargo | Selection steps (early stopping, calibration, meta-learner) left unguarded |
| ML on financial tables | {{c:gu,krauss,fischer,grinsztajn}} | Evidence of small but real signal | Protocols differ; integrity of inputs rarely audited |
| Ensembles and calibration | {{c:wolpert,smith,guo,ovadia}} | Stacking, combination puzzle, calibration | Not tested for 3-class, regime-shifting, calibrated inputs |
| NEPSE studies | {{c:giri,bajracharya,mishra}} | Price and behaviour forecasts | No stated leakage-safe protocol or significance testing |

**Data management for ML.** Production ML has pushed data validation and quality into the foundations of the stack. Breck et al. describe schema-, drift- and skew-based validation of training data at scale {{c:breck}}, and Schelter et al. propose declarative, incrementally computable data-quality checks {{c:schelter}}. Polyzotis et al. survey data-lifecycle challenges in production ML {{c:polyzotis}}, Whang et al. frame data collection and quality as the central lever of data-centric AI {{c:whang}}, and Sculley et al. identify data dependencies as a main source of hidden technical debt {{c:sculley}}. Feature stores standardise how engineers define, version and serve features {{c:orr}}. Our audits follow this tradition, but target a defect class that schema checks cannot see: values that are well-formed and in range yet use information from the future, or labels made wrong by a corporate action.

**Leakage and evaluation of temporal models.** Kaufman et al. formalise leakage in data mining {{c:kaufman}}; Kapoor and Narayanan document its prevalence across ML-based science {{c:kapoor}}; Yang et al. detect it statically in notebooks {{c:yang}}. For dependent data, standard cross-validation can be invalid or optimistic {{c:bergmeir12,bergmeir18,arlot}}, and out-of-sample schemes that respect time order are preferred for non-stationary series {{c:cerqueira}}. For financial labels that look ahead, López de Prado recommends purging training samples whose label window overlaps the test set and embargoing a buffer after it {{c:lopez}}; we adopt both. We add what such guidance omits in practice: the same discipline applied to early stopping, calibration and the meta-learner (Table III).

**Learning from tabular financial data.** Gu, Kelly and Xiu show that tree ensembles and neural networks extract modest but real out-of-sample signal for asset pricing {{c:gu}}; Krauss et al. and Fischer and Krauss compare deep networks, boosted trees and forests for statistical arbitrage {{c:krauss,fischer}}. Large benchmarks find that tree ensembles remain hard to beat on typical tabular data {{c:grinsztajn,shwartz}}. We use the three dominant boosting libraries {{c:xgb,lgbm,cat}} and a Transformer encoder {{c:vaswani}}, and study the focal loss {{c:lin}} as an XGBoost objective for the ambiguous FLAT class.

**Ensembles, calibration and forecast combination.** Stacked generalisation {{c:wolpert,breiman}} learns combination weights, while the forecast-combination literature has long found that simple averages are hard to beat {{c:clemen,timmermann}}. Smith and Wallis explain this "forecast combination puzzle" by the estimation error of the weights, which can outweigh the gain from using the optimal ones when the sample is limited {{c:smith}}. Probabilistic combination additionally requires calibrated inputs: boosted trees and neural networks are often miscalibrated {{c:niculescu,guo}}, and calibration degrades under dataset shift {{c:ovadia}}. Our RQ4 tests the puzzle in a three-class, regime-shifting setting.

**Statistics and multiple testing in finance.** Selecting among many backtested configurations inflates apparent performance {{c:bailey14,bailey,harvey_liu}}, and published predictors weaken after publication {{c:mclean}}. We use the Diebold–Mariano test with the Harvey–Leybourne–Newbold correction {{c:dm,hln}}, a Newey–West variance {{c:newey}}, the exact McNemar test {{c:mcnemar}}, a moving-block bootstrap {{c:kunsch}} and Benjamini–Hochberg control {{c:bh}}, and report the multiclass MCC {{c:matthews,gorodkin,chicco}} next to balanced accuracy.

**Prediction on NEPSE.** Recent NEPSE studies use deep sequence models and graph networks (Table II). From the publication pages we reviewed, they forecast price levels or market behaviour with error metrics, and the pages do not describe a leakage-safe split, probability calibration or a significance test. We cannot judge from abstracts what the full texts do, so we make no claim about their protocols; our contribution is to specify and measure one. NEPSE's price-limit design also matters for data engineering: price limits bound legitimate daily moves {{c:kimrhee}}, which is what makes the band a usable audit for unadjusted corporate actions.

**Summary of the gap.** Existing work validates schemas, detects leakage after the fact, or reports accuracy under protocols that vary from study to study. None of it combines a point-in-time feature store, an audit that is itself validated, a split that guards every selection step including the combiner, and statistical testing of the outcome. Table I maps these lines of work to what each leaves open.

::: {custom-style="TabCap"}
TABLE II. Recent NEPSE prediction studies (information from publication pages; "n.s." = not stated there).
:::

| Study | Task, data | Models | Metrics; validation |
|:--|:--|:--|:--|
| Giri et al. 2025 {{c:giri}} | Price forecast, insurance stocks 2020–24 | LSTM | MAE, MSE, RMSE; n.s. |
| Bajracharya et al. 2025 {{c:bajracharya}} | Market behaviour from visibility graph | GCN, GAT | n.s.; n.s. |
| Mishra 2026 {{c:mishra}} | Price forecast, OHLCV plus FinBERT news sentiment | LSTM, GRU, TCN | MSE, RMSE, MAE, R², MAPE; n.s. |
| **This work** | Next-day 3-class direction, 101 stocks 2012–26 | CatBoost, LightGBM, XGBoost, Transformer, ensembles | Bal. acc., MCC, log loss, ECE; purged walk-forward, 5 seeds, DM, McNemar, bootstrap |

# III. Problem Formulation and Leakage Model

**A. Task.** Let $i \in \{1,\dots,N\}$ index securities and $t \in \mathcal{T}$ trading days. Let $\tilde P_{i,t}$ be the corporate-action-adjusted close and $x_{i,t} \in \mathbb{R}^{d}$ a feature vector that must be computable from information available at the close of day $t$. The forward return is

$$r_{i,t} = \frac{\tilde P_{i,t+1}}{\tilde P_{i,t}} - 1 \quad (1)$$

and the label is UP if $r_{i,t} > \tau$, DOWN if $r_{i,t} < -\tau$ and FLAT otherwise, with $\tau = 0.005$ fixed a priori. A model outputs $\hat p_{i,t} \in \Delta^{2}$ and the predicted class $\hat y = \arg\max_c \hat p_{c}$; its confidence is $\max_c \hat p_{c}$. The estimand is the expected loss on future blocks that were never used to fit or select anything.

**B. Where leakage enters.** The label uses data at $t+1$ by construction, so every step that touches labels is a potential channel. Table III lists the sources we control and how each is tested.

```{=openxml}
<w:p><w:pPr><w:spacing w:before="0" w:after="0"/><w:sectPr><w:type w:val="continuous"/><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="900" w:bottom="1440" w:left="900" w:header="720" w:footer="540" w:gutter="0"/><w:cols w:num="2" w:space="360"/></w:sectPr></w:pPr></w:p>
```

::: {custom-style="TabCap"}
TABLE III. Leakage sources, controls in this work and tests.
:::

| ID | Source | Control | Test or evidence |
|:-:|:--|:--|:--|
| L1 | Feature uses data after $t$ | Backward-looking features; same-day rank normalisation | Truncation-invariance test; IC scan with planted canary |
| L2 | Training labels overlap the test period | Purge $h$ days plus embargo $e$ | Split-gap check (Table IV) |
| L3 | Random k-fold over autocorrelated rows | Chronological blocks only | Protocol experiment (Table VII) |
| L4 | Corporate actions create false labels | Adjusted close; breach audit | Breach counts; labels beyond the band dropped |
| L5 | Selection on test data: early stopping, calibration, hyper-parameters | Embargoed validation tail inside each training window; fixed hyper-parameters | Code path never reads test rows |
| L6 | Combiner fitted on overlapping predictions | Trained on earlier blocks, with purge and embargo | Training-set assertions in unit tests |
| L7 | Statistics fitted on future data | Time-decay and class weights from the training window; FLAT threshold fixed | Unit tests |
| L8 | Whole-pipeline faults | Shuffled-label control | MCC near zero (Section VIII-B) |

```{=openxml}
<w:p><w:pPr><w:spacing w:before="0" w:after="0"/><w:sectPr><w:type w:val="continuous"/><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="900" w:bottom="1440" w:left="900" w:header="720" w:footer="540" w:gutter="0"/><w:cols w:num="1" w:space="360"/></w:sectPr></w:pPr></w:p>
```

# IV. Case-Study Data and Integrity Audits

**A. Dataset.** The panel has one row per (date, security) from 26 February 2012 to 3 November 2026: 148,187 rows, 101 securities, 3,343 trading dates and 72 numeric features plus symbol and sector identifiers. Features cover lagged returns, trend and momentum indicators (SMA distance, RSI, Bollinger %B), volatility, volume, relative strength against sector and market, and microstructure features (trade-size dispersion, volume-profile concentration, broker-anomaly scores) [[verify against the data dictionary]]. Sectors are integer codes 37–41 [[add names]]. The microstructure features are missing for up to 32.5% of rows, 0.57% of rows have zero volume, and 7,223 symbol-days are missing inside the listing period of their security. Tree models handle missing values natively; the Transformer receives zeros. The data were collected by Laganisutra from NEPSE and supplied to the authors free of charge for research use; the provider permits the data to be released with the paper’s artifacts.

Labels are DOWN 39.4%, FLAT 27.7% and UP 33.0%. Always predicting the most frequent training class yields 41.3% accuracy on the test period but a balanced accuracy of exactly 0.333, which is why balanced accuracy and MCC are our primary metrics.

**B. Corporate-action audit.** Let $L = 0.10$ be the daily price band and $\epsilon = 0.005$ a tolerance. A move beyond the band cannot be a genuine trade, so it flags an unadjusted bonus or rights issue or a data error:

$$b_{i,t} = \mathbb{1}\!\left[\left|\frac{P_{i,t}}{P_{i,t-1}} - 1\right| > L + \epsilon\right] \quad (2)$$

We count $\sum b_{i,t}$ on the raw and adjusted series. The same bound applied to the label window, $|r_{i,t}| \le (1+L)^{h}-1+\epsilon$, identifies labels that cannot be real; these are dropped and counted in the second experimental run.

**C. Leakage scan with a canary.** For each feature we compute the information coefficient (IC), the mean over dates of the cross-sectional Spearman correlation between the feature and the forward return. Genuine daily signals rarely exceed $|\mathrm{IC}| \approx 0.1$, so large values indicate future information. A scan is only useful if it can detect leakage, so we plant a canary $x^{\text{can}} = r + \eta$ with noise scaled by the median absolute deviation (a robust choice, since extreme returns inflate the standard deviation) and require it to be flagged.

**D. Truncation invariance.** A feature is point-in-time correct if its value on date $s$ is the same whether the feature code is run on the full data or only on data up to $s$. Any feature value that differs, or exists only when future data is present, is a look-ahead violation. The test catches, for example, a centred moving average.

# V. Method

```{=openxml}
<w:p><w:pPr><w:spacing w:before="0" w:after="0"/><w:sectPr><w:type w:val="continuous"/><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="900" w:bottom="1440" w:left="900" w:header="720" w:footer="540" w:gutter="0"/><w:cols w:num="2" w:space="360"/></w:sectPr></w:pPr></w:p>
```

::: {custom-style="FigureWide"}
![Fig. 1. General architecture. A point-in-time feature store feeds four base learners that share one training recipe; calibrated probabilities are combined and evaluated under the walk-forward protocol.](@@ARCH@@){width=7.1in}
:::

```{=openxml}
<w:p><w:pPr><w:spacing w:before="0" w:after="0"/><w:sectPr><w:type w:val="continuous"/><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="900" w:bottom="1440" w:left="900" w:header="720" w:footer="540" w:gutter="0"/><w:cols w:num="1" w:space="360"/></w:sectPr></w:pPr></w:p>
```

**A. Feature store.** Each feature is converted to a within-date percentile rank,

$$\tilde x_{i,t,k} = \frac{\operatorname{rank}_t\!\left(x_{i,t,k}\right)}{N_t} - \frac{1}{2} \quad (3)$$

where ranks average ties, $N_t$ is the number of securities with a value on day $t$ and missing values become 0. Equation (3) uses only same-day cross-sectional information, removes market-wide level shifts and lets the model compare each stock with its peers.

**B. Base learners.** *CatBoost* {{c:cat}}: depth 7, learning rate 0.04, up to 2,000 iterations, native categorical symbol and sector. *LightGBM* {{c:lgbm}}: 31 leaves, learning rate 0.035, at least 50 samples per leaf. *XGBoost* {{c:xgb}}: histogram trees, depth 4, learning rate 0.035, minimum child weight 20, row and column subsampling 0.8 and 0.7, L2 5.0, L1 0.5, categorical partition splits. *Transformer* {{c:vaswani}}: a 20-day lookback over all features, two pre-norm encoder layers, four heads, model dimension 64, feed-forward 96, dropout 0.2, mean and last-token pooling, padding masks for short histories.

**C. Shared training recipe.** All learners use the same recipe. (i) The last 15% of every training window, separated by an embargo of $e = 25$ days, is a validation slice for early stopping and calibration; the test block is never read. (ii) Training row $n$ receives the weight

$$w_n \propto \frac{1}{n_{y_n}}\; 2^{-a_n/H} \quad (4)$$

where $n_c$ is the number of training rows of class $c$ (class balancing), $a_n$ the age of the row in days relative to the last training date, and $H = 500$ the half-life; weights are rescaled to mean one. (iii) A single temperature $T$, fitted on the validation slice by minimising log loss, rescales the logits as $z/T$ {{c:guo}}.**D. Customised XGBoost objective.** The multiclass focal loss {{c:lin}} with $p = \operatorname{softmax}(z)$ is $\ell_\gamma = -(1-p_y)^{\gamma}\log p_y$. We supply its gradient and a positive diagonal Hessian approximation to XGBoost (the exact Hessian can be negative) and verify the gradient against finite differences. With $\gamma = 0$ the loss reduces to softmax cross-entropy. Section VIII-C ablates each customisation.

**E. Walk-forward protocol.** Order the $T$ distinct dates and let the first 40% be a warm-up used only for training. The remaining dates are split into $K = 6$ consecutive test blocks starting at indices $s_1 < \dots < s_K$. With label horizon $h = 1$ and embargo $e = 25$, the training set and test set of block $k$ are

$$\mathcal{T}^{\text{tr}}_k = \{\,t \le s_k - h - e\,\}, \qquad \mathcal{T}^{\text{te}}_k = \{\,s_k \le t < s_{k+1}\,\} \quad (5)$$

so every training label has resolved before the block starts, with a further embargo that absorbs serial correlation. Blocks 0 and 1 train the combiner and are not scored; blocks 2 to 5 (94,130 stock-days, January 2021 to November 2026) are evaluated. Block 2 is the development block and blocks 3 to 5 the holdout: any configuration chosen after looking at results is confirmed on blocks 3 to 5.

::: {custom-style="TabCap"}
TABLE IV. Walk-forward blocks. The gap between the last training date and the first test date is 26 trading days ($h+e$) in every block.
:::

@@T_BLOCKS@@

**F. Combining learners.** Let $p^{(m)}_n$ be the calibrated probabilities of base learner $m \in \{1,\dots,M\}$, $M = 4$. We compare (a) probability averaging, (b) log-probability averaging and (c) a multinomial logistic stacker, written with the standard softmax:

$$\bar p_n = \frac{1}{M}\sum_{m=1}^{M} p^{(m)}_n \quad (6)$$

$$\hat p_n = \operatorname{softmax}(W z_n + b) \quad (7)$$

where $z_n$ stacks the log-probabilities of the $M$ learners and $W$ is fitted by L2-regularised log loss ($C = 0.1$), and (d) the same stacker with balanced class weights. The stacker for block $k$ trains only on out-of-fold predictions from earlier blocks, with the purge and embargo of Eq. (5).

**Post-hoc decision, disclosed.** Before any results we planned the logistic stacker as the proposed method. After the first full run showed it underperforming (Section VIII-A) we adopted averaging as the primary ensemble and kept the stacker as an ablation. This choice was made after viewing evaluation-period results; Section VIII-A reports what can and cannot be concluded.

# VI. Evaluation Methodology

**A. Predictive and probabilistic metrics.** With confusion matrix $C$, true-class totals $t_k$, predicted-class totals $q_k$, $c = \operatorname{tr}C$ correct and $s$ samples, the multiclass MCC {{c:gorodkin}} summarises the whole confusion matrix in one correlation-like number, and balanced accuracy is

$$\mathrm{BA} = \frac{1}{3}\sum_{k=0}^{2}\frac{C_{kk}}{t_k} \quad (8)$$

Chance level is $\mathrm{BA} = 1/3$ and $\mathrm{MCC} = 0$. We also report macro-F1, log loss, the multiclass Brier score {{c:brier}} and the expected calibration error (ECE), the gap between confidence and accuracy averaged over 15 confidence bins.

**B. Selective prediction.** Predictions are ranked by confidence. Accuracy at coverage $\kappa$ is the accuracy on the most confident $\kappa n$ predictions, and the area under the risk–coverage curve (AURC) {{c:geifman}} is the mean error over all coverage levels (lower is better). We use coverage rather than a fixed 0.80 confidence cut-off because maximum predicted probabilities on this task rarely exceed 0.55.

**C. Statistical comparison.** For models $A$ and $B$ let $d_t$ be the difference of mean log loss on date $t$ over $T$ evaluated dates. The Diebold–Mariano test of the mean difference $\bar d$, with Newey–West variance and the Harvey–Leybourne–Newbold small-sample correction, is compared with a $t$ distribution {{c:dm,hln,newey}}. The exact McNemar test uses the discordant counts $b$ and $c$ of paired correctness {{c:mcnemar}}. Differences in MCC and macro-F1 receive 95% intervals from a moving-block bootstrap over dates (1,000 replicates, block length 20) {{c:kunsch}}. P-values across comparisons are adjusted by Benjamini–Hochberg {{c:bh}}. Results are mean (standard deviation) over five seeds; the standard deviation reflects model randomness, not data uncertainty.

**D. Economic evaluation.** Each day hold an equal-weight long-only portfolio $S_t$ of stocks predicted UP (short selling is unavailable on NEPSE), with weights $w_{i,t} = 1/|S_t|$. With turnover $u_t = \sum_i |w_{i,t}-w_{i,t-1}|$ and round-trip cost $c$ charged as $cu_t/2$ per day, the break-even cost is

$$c^{\star} = \frac{2\,\bar R^{\text{gross}}}{\bar u} \quad (9)$$

where $\bar R^{\text{gross}}$ is the mean daily gross return, $\bar u$ the mean daily turnover, and $c^{\star}$ is the round-trip cost at which average net return is zero. Charging costs on turnover, not on every position every day, matters: a naive implementation overstates costs by an order of magnitude.

# VII. Experimental Setup (NEPSE Case Study)

**Baselines.** *Majority class* (training class frequencies); *persistence* (the class of today's realised move, with probabilities from its empirical conditional distribution in the training window); *momentum-20* (the same for the 20-day return); and multinomial *logistic regression* on the standardised features. **Hyper-parameters** are fixed a priori and not tuned on test blocks. **Software.** All models ran on CPU with XGBoost 3.1.3, LightGBM 4.6.0, CatBoost 1.2.8 and PyTorch 2.10.0 (CPU build) [[add CPU model, cores, RAM, OS, Python and scikit-learn versions]]. The full run (five seeds, ten XGBoost variants and the protocol experiment) took 5,448 s.

# VIII. Case-Study Results (NEPSE)

[[Numbers in this section are from the first full run. Replace with the second run, which drops labels beyond the price band, uses turnover-based costs and reports the holdout split, before submission.]]

**A. Main comparison.** Table V and Table VI give the main results.

::: {custom-style="TabCap"}
TABLE V. Walk-forward test results, blocks 2–5 (94,130 stock-days). Mean over five seeds; digits in brackets are the standard deviation in units of the last digit. Chance: balanced accuracy 0.333, MCC 0.
:::

@@T_MAIN@@

Every learned model exceeds the chance-level balanced accuracy of 0.333, and the best reach 0.38 and an MCC of about 0.07. Averaging the four calibrated probability vectors gives the highest MCC (0.072), the lowest log loss (1.074) and the lowest calibration error (0.014, one third below the best single model, XGBoost at 0.021). On MCC it is not distinguishable from CatBoost (0.071) or the Transformer (0.070), so its benefit lies in probability quality and stability across seeds (standard deviation 0.001 against 0.002 to 0.008). It exceeds the strongest naive baseline, persistence, by 0.016 in MCC and 0.017 in balanced accuracy.

::: {custom-style="TabCap"}
TABLE VI. Accuracy, per-class F1 and AURC (lower is better), mean over five seeds.
:::

@@T_PERCLASS@@

Table VI shows why accuracy alone misleads. The majority baseline reaches the highest accuracy (0.413) by always predicting DOWN, giving F1 of zero on FLAT and UP. The learned models trade a little accuracy for non-zero F1 on every class, and the averaged ensemble has the lowest AURC (0.561), so its confidence ranking is the most informative. FLAT is the hardest class for all models (F1 0.25 to 0.33).

*The learned stacker is worse than every single model.* The logistic stacker on logits reaches an MCC of 0.037, about half that of its own inputs. Its MCC, computed on seed-averaged predictions, is lower than CatBoost's by 0.036 (95% block-bootstrap interval [−0.046, −0.025]), than the mean ensemble's by 0.034 ([−0.043, −0.026]) and than logistic regression on raw features by 0.024 ([−0.036, −0.012]). After Benjamini–Hochberg correction the Diebold–Mariano log-loss test is significant against CatBoost, the Transformer, logistic regression and the mean ensemble, but not against LightGBM or XGBoost; the McNemar test is not significant against XGBoost or CatBoost (adjusted p = 0.89 and 0.054). The evidence is therefore strongest for the MCC and log-loss differences. Balanced class weights recover the MCC (0.071) but not log loss or calibration; the stacker is also below persistence (difference −0.019, interval [−0.036, −0.001]).

*Scope of the claim.* Because averaging was adopted after seeing these results, we claim only that the learned stacker is dominated here, not that averaging has been confirmed as superior. [[Insert holdout-only comparison and pairwise tests for averaging from run 2.]]

**B. Evaluation shortcuts (RQ1).**

::: {custom-style="TabCap"}
TABLE VII. The same XGBoost model under different protocols (seed 0).
:::

@@T_PROTOCOL@@

Random 5-fold validation raises MCC from 0.064 to 0.105, an inflation of about @@INFL@@%, and balanced accuracy from 0.378 to 0.406 (Table VII, Fig. 2). The shuffled-label control yields an MCC of −0.012 and a balanced accuracy below chance, which indicates that the pipeline itself does not leak: with labels destroyed, no signal remains. At a one-day horizon the embargo length has no consistent effect (MCC 0.078, 0.074, 0.061, 0.064, 0.083 for embargoes of 0, 5, 10, 25, 50). The pattern is non-monotone, so we attribute it to different training windows and early-stopping points, not to leakage, and expect the embargo to matter at longer horizons where label windows overlap. All protocol rows use one seed, so differences near 0.01 in MCC are within noise.

::: {custom-style="FigureCol"}
![Fig. 2. Balanced accuracy of the same XGBoost model under different protocols (seed 0). Blue: walk-forward. Red: protocols that leak. Grey: shuffled-label control.](@@PROT@@){width=3.2in}
:::

**C. XGBoost customisations (RQ3).**

::: {custom-style="TabCap"}
TABLE VIII. XGBoost ablation, one switch changed at a time (five seeds; the original configuration is deterministic).
:::

@@T_XGB@@

The customisations are not uniformly beneficial. Class-balanced weights help balanced metrics (MCC 0.066 against 0.047 without). Two choices hurt: removing time-decay weighting raises MCC from 0.066 to 0.085 and removing symbol and sector identifiers raises it to 0.074. A plausible reading is that identifiers let the model memorise per-stock behaviour from earlier regimes and that recency weighting over-fits the latest period, but we did not test these mechanisms. Focal loss with $\gamma = 1$ or 2 improves no metric. Temperature scaling leaves discrimination unchanged and, for XGBoost, slightly raises ECE (0.013 to 0.021). The original unregularised configuration is competitive on MCC (0.071) and ties the best log loss (1.074) with the lowest ECE (0.005). We therefore claim that the customisation effects are measurable switch by switch and mixed, not that they improve XGBoost. [[Run 2 adds a variant with time decay and identifiers removed; confirm it on holdout blocks 3–5 before adopting it.]]

**D. Why does the learned stacker fail (RQ4)?**

::: {custom-style="TabCap"}
TABLE IX. Leave-one-model-out for the logistic stacker (logits).
:::

@@T_STACKABL@@

Removing any single model leaves the stacker's MCC unchanged or lower (Table IX), so the loss is not caused by one bad input. The coefficients are unstable across blocks: the summed absolute weight on the Transformer falls from 3.4 to 1.5 (seed 0) and from 3.6 to 1.8 (seed 1) between blocks 2 and 5, while the stacker's MCC declines from block 2 to block 5 (0.068, 0.064, 0.070, 0.052) and CatBoost's stays between 0.079 and 0.095 (Table X). This is consistent with the forecast-combination puzzle {{c:smith}}: with 23,000 to 85,000 training rows, three classes, a weak signal and regime shifts, the estimation error of 36 weights outweighs any gain from tuning them, while an equal-weight average has no parameters to drift. We did not test other meta-learners (constrained-weight or gradient-boosted), which could behave differently.

**E. Stability across regimes.**

::: {custom-style="TabCap"}
TABLE X. MCC by walk-forward test block (mean over five seeds). Block 2 covers 2021–22, block 3 2022–23, block 4 2023–25 and block 5 2025–26.
:::

@@T_REGIME@@

The four base models stay positive in all four blocks (MCC 0.065 to 0.098) and above persistence in each, so the signal is not an artefact of one regime. [[Add per-block results for the averaging ensemble and a sector breakdown with sector names from run 2.]]

**F. Calibration and selective prediction.** Maximum predicted probabilities are low: even among the 5% most confident predictions the threshold is about 0.55, so a fixed 0.80 flag would never fire. For the logistic stacker (the only method for which this curve was computed in the first run), accuracy rises from 0.396 on all predictions to 0.430 on the most confident 20% and 0.433 on the most confident 25%. This is above the unconditional accuracy but only modestly above the 0.413 obtained by always predicting the majority class, so we do not present confidence-filtered accuracy as a trading-grade signal. [[Insert accuracy at 10, 20 and 30% coverage, the risk–coverage curve and a reliability diagram for the averaging ensemble from run 2.]]

**G. Data-integrity findings (RQ2).** The corporate-action audit finds 219 moves beyond the band in the unadjusted close and 85 in the series treated as adjusted, across 34 securities. The adjustment therefore removed 61% of breaches but left a substantial remainder: some bonus or rights issues are missing or only partly adjusted, and each creates a forward-return label that is wrong by up to the adjustment factor. In the second run these labels are dropped and counted [[report number dropped and the change in each metric]].

The information-coefficient scan flags no feature at the first run's threshold of 0.25. The strongest absolute coefficients are −0.095 (relative strength against sector) and −0.090, −0.087, −0.083, −0.077 (return lags 2 to 5): short-horizon reversal and relative-strength effects, which are plausible. The scan alone is weak evidence, because the planted canary was not flagged in the first run (IC 0.24 against the 0.25 threshold): extreme returns had inflated the noise scale used to construct it. We reproduced the failure, replaced the noise scale with a median-absolute-deviation estimate and lowered the threshold to 0.15, still above every real feature; the corrected audit flags the canary (IC 0.59 in a test with heavy-tailed returns). [[Report the corrected canary result on the real data from run 2.]]

**H. Economic evaluation.** [[To be completed from run 2. The first run's trading numbers are invalid: its cost model charged a full round-trip cost every day regardless of whether holdings changed, giving returns of about −78% per year for every method. Report gross and net returns, turnover and the break-even cost $c^{\star}$ from Eq. (9) for several illustrative cost levels, and cite the current NEPSE commission, regulator-fee, depository-charge and capital-gains-tax schedule if a specific net figure is quoted.]] A daily one-day-ahead signal turns the portfolio over frequently (about 1.5 times per day in a synthetic test of the corrected code), so we expect the net result to be dominated by costs unless $c^{\star}$ exceeds the actual cost; we will report it either way.

**I. Efficiency.**

::: {custom-style="TabCap"}
TABLE XI. Mean cost per walk-forward fold (training 65,474 rows, test 19,674 rows; CPU only) [[add CPU and RAM]].
:::

@@T_EFF@@

The boosted-tree models fit in 3 to 14 seconds per fold and serve hundreds of thousands of rows per second; the Transformer is about 35 times slower to fit and 20 times slower at inference than LightGBM and XGBoost. Because its MCC (0.070) is comparable to CatBoost's (0.071), the ensemble's cost is dominated by it. [[Add incremental daily-update latency and a synthetic scale-up test over securities and dates: the efficiency experiments a data-engineering audience will expect.]]

# IX. Discussion

**Implications for data engineers.** The results suggest five practices for temporal ML data. (i) Treat the split, not the model, as the unit of design: purge by the label horizon and embargo beyond it, for base learners and combiners alike. (ii) Control selection leakage: early stopping, calibration and ensemble weights must be fitted on data that precedes the test block. (iii) Validate values against domain invariants, such as a price band, because well-formed values can still be wrong. (iv) Validate the validator: a planted canary exposed a defect in our own leakage scan. (v) Report effect sizes with intervals and the chance level; an MCC of 0.07 is a different claim from "74% accuracy".

**Threats to validity.** *Internal:* the first full run contained mislabelled rows from unadjusted corporate actions, quantified here and removed in the second run; the primary ensemble was chosen after viewing results, which the holdout confirmation mitigates but does not remove. *Construct:* the 0.5% FLAT threshold and the one-day horizon are fixed, and sensitivity to them, to market-relative labels and to quantile-based thresholds is supported by the code but not reported; direction accuracy is not profit. *Statistical:* the protocol experiment uses one seed; differences of about 0.01 in MCC are comparable to seed and block variation, so we do not rank models whose intervals overlap; seed variance understates data uncertainty. *External:* one market and one data provider; Transformer and tree models use default configurations, so the comparison reflects defaults and not tuned optima.

# X. Conclusion and Availability

We presented a general leakage-safe data-preparation and evaluation pipeline for direction prediction from financial time series and used the NEPSE case study to show that evaluation design and data integrity, more than model choice, determine what can be claimed. On NEPSE, calibrated averaging of four learners gives the best-calibrated probabilities and an MCC of 0.072; a learned stacker does not improve on its inputs; and random cross-validation overstates the result by about 65%. Future work includes longer horizons, market-relative labels, other frontier markets and an incremental streaming version of the feature store.

**Artifacts.** All artifacts needed to reproduce the reported results, namely code, configuration, 17 unit tests plus the incremental-update test, the audit and efficiency scripts, the experiment runner that regenerates every table and figure, per-run logs and the released predictions, are in a public GitHub repository at [[GitHub URL; archive a release with a Zenodo DOI]]. The NEPSE data, supplied by Laganisutra for research use, are included in the repository with the provider’s permission, together with the schema, the data dictionary and the scripts that build every table from it.

**AI-use disclosure.** Generative AI was used as follows. Anthropic’s Claude (Claude Sonnet 5.5 through Claude Code) wrote the experiment code (feature-store audits, walk-forward pipeline, model wrappers, evaluation and efficiency scripts) and its unit tests, produced Fig. 1 and the scripts that generate Tables I–XI from result files, drafted and revised the text and equations of every section, and helped locate and summarise the cited literature. OpenAI’s ChatGPT ran the experiment scripts on the authors’ data and returned the result files [[confirm the ChatGPT model and version]]. The authors specified the study, supplied the data, reviewed the code and results, verified every number against the result files and every reference against its source, and take responsibility for the correctness and originality of the paper. No AI system is listed as an author.

**Acknowledgments.** The authors thank Laganisutra for providing the NEPSE data for research. [[Funding, or “none”.]]

# References

@@REFS@@
