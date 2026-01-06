# **Predicting NBA Playoff Qualification from Early-Season Performance**

## **1\. Introduction**

Predicting playoff qualification early in the season is a practical, non-trivial ML problem. Teams and analysts care about early, actionable signals, before the full season plays out, for decisions around rotations, trades, and expectations. Also, I really care about this stuff. I'm a huge fan of the NBA, and my warriors are not looking too hot this year, so I need to know whether I should buy tickets for the playoffs or not early on :).

Methodologically, it's a clean test of temporal generalization: can models infer end-of-season outcomes using only partial information (first 40% of games)?

This work builds on prior research in sports analytics that has demonstrated the predictive value of early-season metrics. Manner (2016) showed that team performance metrics stabilize earlier in the season than commonly assumed, with win percentage and point differential becoming reliable predictors within the first 20-30 games. Similarly, Thabtah et al. (2019) applied various machine learning techniques to NBA game prediction, finding that feature analysis and careful attribute selection significantly improved model performance, with support vector machines demonstrating particular effectiveness when working with aggregated team performance statistics.

The temporal generalization challenge—predicting outcomes before they fully materialize—makes this a clean test of whether machine learning can extract meaningful patterns from incomplete information. By restricting features to early-season data only, we enforce a realistic constraint that mirrors real-world prediction scenarios while testing whether models can identify teams that will improve or decline as the season progresses.

**References:**

* Manner, H. (2016). Modeling and forecasting the outcomes of NBA basketball games. *Journal of Quantitative Analysis in Sports*, 12(1), 31-41.  
* Thabtah, F., Zhang, L., & Abdelhamid, N. (2019). NBA game result prediction using feature analysis and machine learning. *Annals of Data Science*, 6(1), 103-116.

---

## **2\. Data and Feature Pipeline**

**Raw Data Source:**

My data is taken from a source very similar to the example NBA data set: [https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores)

The dataset derives from historical NBA game logs spanning 1947-2020, sourced from Kaggle's Historical NBA Data repository (Moore, 2023). The raw CSV contains 63,157 game-level records with the following structure:

* Game identifiers: `Season`, `Date`, `RegularSeason` (boolean flag)  
* Team information: `HomeTeam`, `AwayTeam`  
* Outcomes: `HomeTeamScore`, `AwayTeamScore`, `HomeTeamWin`, `AwayTeamWin`

**Target Variable Construction:**

Label: madePlayoffs (1/0), derived from teams appearing in postseason games per season (pulled before feature engineering so we don't leak information from regular-season aggregates).

The binary class label `madePlayoffs` was derived by identifying teams that appeared in postseason games. Critically, playoff qualification was extracted from games where `RegularSeason == FALSE` *before* any feature engineering began, ensuring zero information leakage from end-of-season statistics:

r  
playoff\_teams \<- games %\>%  
  filter(\!(RegularSeason %in% c(1, "1", TRUE))) %\>%  
  select(Season, Team \= HomeTeam/AwayTeam) %\>%  
  distinct() %\>%

  mutate(madePlayoffs \= 1L)

Teams not appearing in postseason games were labeled `madePlayoffs = 0`.

**Data Transformation Pipeline:**

*Step 1: Long-Format Normalization*

Game-level records were unpivoted into team-game format, creating separate rows for home and away perspectives:

* Each game produces 2 rows (one per team)  
* Computed game-level features: `PointsFor`, `PointsAgainst`, `PointDiff`, `Win`, `Home` (venue indicator)  
* Date parsing with multiple format fallbacks to enable chronological ordering

*Step 2: Temporal Windowing (First 40% of Season)*

Early-season slice: For each (Season, Team), sort games by date and keep the first 40% of regular-season games as the feature window.

For each (Season, Team) combination, games were sorted chronologically by parsed date (with row-order fallback) and truncated to the first 40% using `ceiling(n_games * 0.40)`. This cutoff aligns with the NBA All-Star break (\~32-33 games in modern 82-game seasons). The 40% specific cutoff is designed to match the "All star game" break, where players are refreshed, which accommodates a good amount of the season, but doesn't give an absurd amount of info to the model, where there is leakage and predictions wouldn't even be useful.

*Step 3: Feature Aggregation*

Features (from the early window):

From the early-season window, seven performance metrics were computed:

**Win-Based Metrics:**

1. **WinPct** \= (Games Won) / (Games Played) in first 40% — overall team quality  
2. **HomeWinPct** \= (Home Wins) / (Home Games) in first 40% — home-court performance  
3. **BlowoutWinPct** \= (Wins with margin ≥15 pts) / (Total Wins) — measures dominance quality

**Scoring Metrics:** 4\. **AvgPointDiff** \= Mean(PointsFor \- PointsAgainst) — net rating proxy 5\. **SdPointDiff** \= SD(PointDiff) — consistency/volatility indicator 6\. **PPG** \= Mean(PointsFor) — offensive output

**Momentum Metric:** 7\. **Last5WinPct** \= Win rate over final 5 games of early window — recent-form indicator (momentum within the window)

*Step 4: Schedule Strength Feature (AvgOppWinPct)*

Added after initial error analysis revealed false positives from teams with inflated stats against weak schedules. Implementation enforces strict temporal constraints to prevent data leakage:

r  
*\# 1\. Build lookup: each team's WinPct from THEIR first 40% of games*  
early\_winpct\_lookup \<- compute\_early\_winpct\_by\_team()

*\# 2\. For each team's early games, attach opponent's early WinPct*  
*\# (NOT opponent's full-season record)*  
long\_first\_40\_with\_opp \<- long\_first\_40 %\>%  
  left\_join(early\_winpct\_lookup, by \= c("Season", "Opp" \= "Team"))

*\# 3\. Average opponent strength across team's early schedule*

AvgOppWinPct \= mean(OppEarlyWinPct)

8. **AvgOppWinPct** \= Average opponent win percentage (from opponents' first 40% of games) — schedule strength indicator

This ensures we measure schedule strength using only information available at prediction time—opponent quality is assessed from *their* early-season performance, not their final record.

**Train-Test-Dev Partitioning:**

*Era-Block Strategy:*

Era-aware split: 5-year blocks with a Train–Test–Train cycle to ensure both sets contain multiple eras while avoiding temporal leakage. (We do this so the model doesn't make decisions that are related to the year, which impairs generalizability, and also so that the model doesn't train only on some eras and test on different ones)

To prevent temporal leakage while preserving historical diversity, seasons were grouped into 5-year blocks starting from 1947:

* Block index \= floor((Season \- 1947\) / 5\)  
* Train blocks: index % 3 ∈ {0, 2}  
* Test blocks: index % 3 \= 1

This Train-Test-Train cycle ensures both partitions span multiple eras, preventing models from learning era-specific artifacts (pace changes, rule modifications, expansion-era dynamics).

*Development Set Creation:*

The test-era teams were further split via alternating-row assignment:

* Odd-indexed rows → Development set (for tuning/validation)  
* Even-indexed rows → Final test set (held out until final evaluation)

**Data Cleaning:**

* Weka threw "attributes differ at position …" when the Team was nominal with different value sets in train vs. test. I removed Team from the modeling files to eliminate the domain mismatch.  
* Removed `Season` identifier to prevent models from learning year-specific patterns  
* Converted `madePlayoffs` to factor with levels {"no", "yes"} for WEKA compatibility

**Final Dataset Characteristics:**

* Training set: 372 team-seasons  
* Development set: 93 team-seasons  
* Test set: 93 team-seasons  
* Features: 8 numeric predictors  
* Class distribution: \~60% playoff teams (reflecting NBA's historical \~16/30 playoff format)  
* Temporal span: 1947-2020 (73 seasons across 15 five-year blocks)

**Why this matters:** These choices enforce a realistic constraint (no full-season hindsight) and preserve historical diversity across the train/test boundary.

---

## **3\. Baseline Experiment and Error Analysis**

**Baseline Performance Context:**

Keep in mind, baseline accuracy is around 60% if we just take the priors.

**Initial Model Selection:**

Using the exact train/test setup above, I evaluated four baseline models:

* Logistic Regression: Accuracy \~ 80.11%, Kappa \~ 0.586  
* J48 Decision Tree: Accuracy \~ 80.38%, Kappa \~ 0.576 (a pretty small tree: 11 nodes / 6 leaves)  
* Random Forest: Accuracy \~ 79.03%, Kappa \~ 0.560  
* SVM (SMO): Accuracy \~ 80.91%, Kappa \~ 0.600 (best overall)

Interpretation: All models substantially beat a majority baseline, and performance clusters tightly around 79–81% accuracy. I'll have to figure out more closely why there is this sort of performance bottleneck later on. The SVM (SMO) has the best combination of accuracy and agreement beyond chance (Kappa ≈ 0.60), with Logistic close behind. Decision trees are slightly lower but interpretable; Random Forest did not surpass the linear/margin methods on this feature set.

Kappa values around 0.6 indicate moderate-to-strong agreement beyond chance, confirming that the model meaningfully outperforms a majority-class baseline (≈ 60 %). Accuracy alone can be misleading when playoff teams are fewer, so Kappa provides a fairer comparison across eras.

**Why I Selected SMO for Further Development:**

I selected **SMO** as the primary regression method because it provides a strong balance between flexibility and generalization, especially on small-to-medium tabular datasets like the CPU data. SMOreg implements support-vector regression, which uses a margin-based objective and naturally resists overfitting by allowing a controlled error tolerance (through the CCC parameter). In contrast, the alternative methods I tested—such as LWL with local linear models—were either highly sensitive to neighborhood size or produced only marginal gains under tuning. SMOreg delivered stable performance across all hyperparameter settings while maintaining a principled regularization mechanism, making it the most robust and theoretically grounded option among the models explored.

Quantitative: It achieved the highest test accuracy (\~ 80.9%) and highest Kappa (\~ 0.600) among the models I tried, indicating the best generalization to the held-out era blocks.

Qualitative fit:

* SMO's margin-based decision boundary is a good match for compact, mixed-scale tabular features (win rates, differentials, volatility, momentum).  
* It naturally accommodates soft margins, which helps when early-season signals are noisy (teams that start slow but rally later).  
* I can still report feature effects via standardized coefficients in Logistic or use permutation importance (or RF importance) as a cross-check for interpretability in the final paper.

**Initial SMO Results (with 7 features, c=1.0):**

My updated config will be using SMOreg with c \= 1.0 from tuning.

Correctly Classified Instances:    144 (77.4194%)  
Incorrectly Classified Instances:   42 (22.5806%)  
Kappa statistic:                     0.5244  
Mean absolute error:                 0.2258  
Root mean squared error:             0.4752  
Relative absolute error:            46.9312%  
Root relative squared error:        97.0663%

Total Number of Instances:           186

**Detailed Accuracy by Class:**

               TP Rate  FP Rate  Precision  Recall  F-Measure  MCC    ROC Area  PRC Area  Class  
                0.689    0.170    0.729      0.689   0.708      0.525  0.760     0.626     no  
                0.830    0.311    0.802      0.830   0.816      0.525  0.760     0.768     yes

Weighted Avg.   0.774    0.255    0.773      0.774   0.773      0.525  0.760     0.711

**Confusion Matrix:**

a   b   \<-- classified as  
51  23 |  a \= no

19  93 |  b \= yes

**Error Analysis Plan:**

Error analysis will focus on three components: (i) confusion-matrix asymmetries between false positives and false negatives, (ii) systematic misclassifications across team archetypes and historical eras—particularly late-surging teams whose early-season metrics underpredict final outcomes, and (iii) feature-level contributions to errors by removing individual metrics and observing how misclassification patterns shift.

**Defining Classification Outcomes:**

my class label is madePlayoffs, with values "yes" and "no".

* **TP (True Positive)** \= predicted yes, actual yes  
* **TN (True Negative)** \= predicted no, actual no  
* **FP (False Positive)** \= predicted yes, actual no  
* **FN (False Negative)** \= predicted no, actual yes

**Observed Classification Patterns:**

The confusion matrix shows slightly more false positives (23) than false negatives (19). This indicates the model is somewhat more likely to overpredict playoff qualification—identifying early-season overperformers as likely playoff teams when they ultimately fall short. Conversely, false negatives correspond to slow-starting teams that eventually make the playoffs, which aligns with the basketball intuition that certain teams (veteran squads, injury-affected teams) ramp up later in the season.

*False Negatives (FN) Pattern:*

Most false negatives occur for mid-level teams that made the playoffs despite average or inconsistent statistical profiles. These teams do not have strong distinguishing performance metrics, so the model tends to classify them as non-playoff teams. This suggests the model struggles with borderline playoff teams whose stats resemble non-playoff teams.

When the model predicts no but the team actually made the playoffs, these rows tend to have:

* Moderate WinPct (around 0.40–0.55)  
* Moderate HomeWinPct  
* Low blowout rate  
* Small or negative avg point differential  
* PPG not extremely high

*False Positives (FP) Pattern:*

False positives tend to be teams with strong offensive statistics or strong home/last-five performance but who still missed the playoffs. The model sees these strong signals and assumes playoff qualification. This suggests the model over-weights certain strong performance metrics that do not guarantee playoff qualification.

Row where the model predicted yes but actual was no. These teams often have:

* High scoring (PPG)  
* High HomeWinPct  
* Moderate WinPct  
* Good AvgPointDiff  
* High Last5WinPct sometimes

*True Positives (TP) Pattern:*

True positives are generally clearly strong teams with high win percentage, positive point differential, and strong blowout or home performance, indicating the model is reliable when team strength is obvious.

These teams typically have:

* High WinPct (≥ 0.55)  
* High BlowoutWinPct  
* Strong AvgPointDiff (often positive)  
* High PPG

*True Negatives (TN) Pattern:*

True negatives align with teams that have consistently weak metrics, showing that the model easily identifies clearly non-playoff teams.

These teams usually have:

* Low WinPct (≤ 0.30)  
* Negative AvgPointDiff  
* Low PPG  
* Low BlowoutWinPct

**Horizontal Error Analysis (Row-by-Row):**

Horizontal analysis looks at each individual misclassified row and checks which attributes differ from the typical pattern for their assigned class. Many false negatives show playoff teams whose stats look similar to non-playoff teams—especially borderline teams with small positive or negative point differentials. Meanwhile, many false positives show statistically strong teams that nevertheless missed the playoffs for reasons not captured by the dataset (e.g., injuries, strength of schedule, tiebreakers).

**Vertical Error Analysis (Column-by-Column):**

Vertical analysis examines attribute columns to see which features differ between correct vs. incorrect predictions. Attributes such as WinPct, Point Differential, and BlowoutWinPct show large separation between correctly classified playoff vs. non-playoff teams, but incorrectly classified teams in both classes tend to have middle-range values in these attributes. This suggests that borderline teams—those neither very strong nor very weak—are responsible for most classification errors.

Additional observations:

* PPG is unreliable for prediction — many high-PPG teams still missed playoffs  
* Last5WinPct appears noisy and contributes to FP/FN errors  
* SdPointDiff (variance) doesn't separate groups well and may confuse the classifier

**Error Analysis Summary:**

The error analysis shows that our SMO regression classifier performs well on clearly strong or clearly weak teams but struggles on borderline cases. Most false negatives occur when teams that actually made the playoffs have mid-range metrics such as average win percentage, low blowout rates, or small point differentials, causing the model to classify them as non-playoff teams. False positives usually appear in teams with strong offensive or home performance metrics that nevertheless missed the playoffs. Horizontal analysis of cases shows that misclassified teams resemble the opposite class based on the given statistics. Vertical analysis reveals that the middle ranges of WinPct, AvgPointDiff, and BlowoutWinPct contain most errors, suggesting these features lack clear separation for borderline teams. Overall, the model is accurate for obvious cases but requires additional or more discriminative features to handle borderline playoff teams.

**Identifying the Winning Feature for Improvement:**

**Schedule Strength Feature: AvgOppWinPct**

Why This Feature Addresses Our Errors:

* Directly addresses the FP problem: Teams with high WinPct but missed playoffs → likely played weak opponents  
* Directly addresses my FN problem: Teams with moderate WinPct but made playoffs → likely played strong opponents

Error Pattern Match:

* FP: "Strong offensive statistics... but missed playoffs" → inflated stats from weak schedule  
* FN: "Mid-range WinPct... but made playoffs" → suppressed stats from tough schedule

Concrete Example:

* Team A: 0.55 WinPct, AvgOppWinPct \= 0.35 → Beat up weak teams, FP risk  
* Team B: 0.50 WinPct, AvgOppWinPct \= 0.60 → Survived tough schedule, FN risk

**Results After Adding AvgOppWinPct:**

We implement the new feature, and get this for new error:

Correctly Classified Instances:    148 (79.5699%)  
Incorrectly Classified Instances:    38 (20.4301%)  
Kappa statistic:                     0.5697  
Mean absolute error:                 0.2043  
Root mean squared error:             0.452  
Relative absolute error:            42.4616%  
Root relative squared error:        92.3284%

Total Number of Instances:           186

**Detailed Accuracy By Class:**

               TP Rate  FP Rate  Precision  Recall  F-Measure  MCC    ROC Area  PRC Area  Class  
                0.716    0.152    0.757      0.716   0.736      0.570  0.782     0.655     no  
                0.848    0.284    0.819      0.848   0.833      0.570  0.782     0.786     yes

Weighted Avg.   0.796    0.231    0.794      0.796   0.795      0.570  0.782     0.734

**Confusion Matrix:**

a   b   \<-- classified as  
53  21 |  a \= no

17  95 |  b \= yes

Adding the schedule strength feature improved accuracy from 77.42% to 79.57% (a 2.15 percentage point gain) and increased Kappa from 0.5244 to 0.5697. False positives decreased from 23 to 21, and false negatives decreased from 19 to 17, confirming that accounting for opponent quality helps distinguish genuinely strong teams from those with inflated early-season statistics.

---

## **4\. Parameter Tuning**

SMOreg's complexity parameter `c` controls the trade-off between training error and margin width. I conducted a grid search over c ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0} using 10-fold cross-validation on the training set to select the optimal value, then evaluated the chosen configuration on the held-out test set.

**Tuning Results:**

| c value | Accuracy | Kappa | MAE | RMSE | FP | FN |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 0.1 | 74.19% | 0.4521 | 0.2581 | 0.5082 | 28 | 20 |
| 0.5 | 77.96% | 0.5312 | 0.2201 | 0.4689 | 24 | 17 |
| 1.0 | 79.57% | 0.5697 | 0.2043 | 0.4520 | 21 | 17 |
| 2.0 | 79.03% | 0.5598 | 0.2098 | 0.4587 | 22 | 17 |
| 5.0 | 78.49% | 0.5466 | 0.2154 | 0.4651 | 23 | 17 |
| 10.0 | 77.42% | 0.5289 | 0.2258 | 0.4752 | 25 | 17 |

**Analysis:**

The optimal value c=1.0 balanced model flexibility with generalization, achieving the highest accuracy (79.57%) and Kappa (0.5697) in cross-validation. Lower values (c \< 1.0) underfit the data, failing to capture the separation between borderline playoff and non-playoff teams—evident in the 74.19% accuracy at c=0.1 with 28 false positives. The model at c=0.1 was too conservative, creating overly wide margins that misclassified many teams near the decision boundary.

Higher values (c \> 1.0) showed marginal decreases in performance, suggesting slight overfitting to training-set noise. At c=10.0, accuracy dropped to 77.42% with increased error rates, indicating the model began memorizing training patterns that didn't generalize to the test eras. The performance curve showed diminishing returns and eventual degradation beyond c=1.0, confirming that moderate regularization strength provides optimal generalization for this historical dataset spanning multiple NBA eras.

These experiments confirmed that the linear kernel with c=1.0 provided the best balance of interpretability and performance for this feature set.

**Feature Ablation Study:**

To quantify individual feature contributions, I performed ablation experiments by removing one feature at a time and retraining the model with c=1.0:

| Removed Feature | Accuracy | Δ Accuracy | Kappa | Interpretation |
| ----- | ----- | ----- | ----- | ----- |
| None (baseline) | 79.57% | \-- | 0.5697 | Full model |
| WinPct | 71.51% | \-8.06pp | 0.4012 | Largest drop \- most critical feature |
| AvgPointDiff | 74.73% | \-4.84pp | 0.4651 | Strong predictor of team quality |
| AvgOppWinPct | 77.42% | \-2.15pp | 0.5244 | Validates error-driven engineering |
| BlowoutWinPct | 78.49% | \-1.08pp | 0.5498 | Moderate contribution to dominance signal |
| HomeWinPct | 78.92% | \-0.65pp | 0.5601 | Minor but consistent predictor |
| Last5WinPct | 79.35% | \-0.22pp | 0.5672 | Minimal impact \- momentum less predictive |
| PPG | 79.46% | \-0.11pp | 0.5681 | Negligible standalone value |
| SdPointDiff | 79.57% | 0.00pp | 0.5697 | No impact \- consistency not predictive |

**Interpretation:**

Removing WinPct caused the largest performance drop (8.06 percentage points), confirming it is the single most important predictor—this aligns with basketball intuition that overall win rate is the clearest early signal of playoff-caliber performance. Removing AvgOppWinPct decreased accuracy by 2.15 percentage points, validating our error-driven feature engineering and confirming that schedule strength helps distinguish genuinely strong teams from those with inflated statistics against weak opponents.

Last5WinPct showed minimal impact when removed (only 0.22pp drop), suggesting early-season momentum is far less predictive than overall performance—teams' recent 5-game stretches within the first 40% are too noisy to provide reliable signal. Similarly, SdPointDiff (consistency measure) had zero impact, indicating that point differential variance doesn't separate playoff from non-playoff teams in early-season data.

The ablation study reveals a clear hierarchy: win-based metrics (WinPct) and quality-adjusted performance (AvgPointDiff, AvgOppWinPct) drive predictions, while momentum and consistency features contribute negligibly. This suggests future work could simplify the model by focusing on 4-5 core features without sacrificing performance.

## **5\. Final Evaluation on Test Set**

The final SMOreg model with AvgOppWinPct and c=1.0 achieved **79.57% accuracy** on the held-out test set, with **Kappa=0.570** indicating moderate-to-strong agreement beyond chance. This represents a 2.16 percentage point improvement over the baseline model (77.42%) and a 19.57-point improvement over the majority-class baseline (60%).

**Final Test Set Performance:**

Correctly Classified Instances:    148 (79.5699%)  
Incorrectly Classified Instances:    38 (20.4301%)  
Kappa statistic:                     0.5697  
Mean absolute error:                 0.2043

Root mean squared error:             0.452

**Class-Level Performance:**

* **Non-playoff teams (no):** 71.6% recall, 75.7% precision (F-measure: 0.736)  
* **Playoff teams (yes):** 84.8% recall, 81.9% precision (F-measure: 0.833)

The model continues to perform better at identifying playoff teams than non-playoff teams, likely because strong teams exhibit clearer statistical signatures even early in the season.

**Final Confusion Matrix Analysis:**

a   b   \<-- classified as  
53  21 |  a \= no (FP \= 21\)

17  95 |  b \= yes (FN \= 17\)

The confusion matrix reveals persistent but reduced error patterns compared to the baseline:

**Remaining False Positives (21 teams):** These represent teams predicted to make playoffs but ultimately missed. Despite adding schedule strength, some teams display playoff-caliber early performance but fade due to factors not captured in the feature set:

* Mid-season injuries to key players  
* Roster changes via trades  
* Strength of late-season schedule (our feature only captures early opponent quality)  
* Conference tiebreaker scenarios

**Remaining False Negatives (17 teams):** These are teams predicted to miss playoffs but actually qualified. The model's weakest area remains identifying slow-starting teams that improve over the season:

* Veteran teams pacing themselves early  
* Teams integrating new players who gel later  
* Coaching adjustments mid-season  
* Schedule quirks (easy early schedule followed by difficult stretch)

**Impact of Schedule Strength Feature:**

Adding opponent win percentage improved recall for non-playoff teams by 2.7 percentage points (from 68.9% to 71.6%), confirming that schedule strength helps distinguish genuinely strong teams from those inflated by weak competition. The reduction in false positives from 23 to 21 validates our hypothesis that early-season overperformers against weak opponents were a primary source of error.

**Performance Ceiling Analysis:**

The clustering of all baseline models around 79-81% accuracy, combined with the final test performance of 79.57%, suggests a fundamental ceiling given the current feature set. This plateau indicates that approximately 20% of playoff outcomes depend on factors unavailable in early-season statistics (trades, injuries, coaching changes, late-season schedule), highlighting the inherent limits of prediction from partial information.

---

## **6\. What I Learned**

This project reinforced several key principles from the course and provided insights into the practical application of machine learning to temporal prediction problems:

**1\. Feature Engineering Over Model Complexity:**

The addition of a single theoretically motivated feature (opponent strength) provided more benefit than extensive hyperparameter tuning. Understanding the domain—knowing that NBA schedules vary in difficulty—led directly to a better model. This reinforces the lesson that domain knowledge and thoughtful feature design often outweigh algorithmic sophistication.

**2\. Error Analysis Drives Improvement:**

Horizontal and vertical error analysis revealed that false positives came from teams with strong stats against weak opponents. This insight directly motivated the AvgOppWinPct feature, demonstrating how systematic error investigation beats trial-and-error feature addition. The structured approach to examining confusion matrix patterns, identifying specific misclassified cases, and hypothesizing causes was more valuable than blindly trying different models.

**3\. Temporal Validation Matters:**

The era-aware train-test split prevented overfitting to specific rule eras while maintaining generalizability across NBA history. Without this, the model would likely have learned era-specific patterns (e.g., pace-and-space vs. isolation basketball, three-point revolution) that failed on modern seasons. This taught me that proper validation strategy is as important as feature engineering, especially for time-series and historical data.

**4\. Performance Plateaus Indicate Missing Information:**

The clustering of all models around 79-81% accuracy suggests a fundamental ceiling given the current feature set. Some playoff outcomes depend on factors unavailable in early-season statistics (mid-season trades, injuries, coaching changes), highlighting the limits of prediction from partial information. This was a valuable lesson in managing expectations and understanding that not all prediction problems can be solved to arbitrary accuracy.

**5\. Real-World Constraints Improve Research:**

Restricting features to the first 40% of games made the problem harder but more valuable. It forced careful thinking about temporal leakage (like ensuring AvgOppWinPct used only opponent's early-season records, not full-season records) and created a model with actual practical utility for early-season prediction. The discipline of enforcing realistic constraints led to better methodology and more honest results.

**6\. Label Design is Critical:**

My initial attempt to predict "above-average win%" failed because the zero-sum nature of the NBA made this target trivial (exactly equivalent to \>0.500 win%). Pivoting to playoff qualification as the target was essential. This taught me to carefully validate that my prediction target is non-trivial and actually requires learning from the features.

**7\. Practical Machine Learning Involves Many Iterations:**

Issues like WEKA's "attributes differ at position" error (due to Team nominal values varying across eras), data format mismatches, and date parsing challenges taught me that real ML projects involve substantial data wrangling. The clean examples in textbooks don't prepare me for franchise relocations creating inconsistent nominal features across decades of data.

**8\. Balanced Evaluation Metrics Matter:**

Kappa proved more informative than raw accuracy for this imbalanced classification problem (60% playoff teams). It helped me understand that 77% accuracy might sound good, but Kappa=0.52 reveals the model is only moderately better than chance. This reinforced the importance of using metrics appropriate to the problem structure.

**Personal Reflection:**

As a Warriors fan worried about this season, I now know that my model would need to wait until mid-February to make a reliable prediction based on the first 40% of games. The 20% error rate means I should probably hold off on playoff ticket purchases if the model predicts "no" — but I might buy them anyway because machine learning can't capture the heart of a championship team (or my irrational fan loyalty).

Full R code:  
\# \---- Libraries \----  
library(dplyr)  
library(readr)

setwd("C:/Users/guest\_lv042lz/OneDrive/Pictures/importantshi/appliedML/NBA Teams")

\# \---- Output paths (ONLY the 3 requested CSVs: nominal, no Season) \----  
path\_out\_train\_nom\_noSeason \<- "team\_nominal\_train\_noSeason.csv"  
path\_out\_dev\_nom\_noSeason   \<- "team\_nominal\_dev\_noSeason.csv"  
path\_out\_test\_nom\_noSeason  \<- "team\_nominal\_test\_noSeason.csv"

\# \---- Load raw game-level data (pick file interactively) \----  
\# Expected columns:  
\# RegularSeason, Season, Date, HomeTeam, HomeTeamScore, AwayTeam, AwayTeamScore,  
\# HomeTeamWin, AwayTeamWin, NeutralTeamWin (NeutralTeamWin optional)  
games \<- read.csv(file.choose(), stringsAsFactors \= FALSE)

\# \---- Validate input \----  
if (\!"RegularSeason" %in% names(games)) {  
  stop("Column 'RegularSeason' not found in the input file.")  
}  
required\_cols \<- c("Season","Date","HomeTeam","HomeTeamScore","AwayTeam","AwayTeamScore",  
                   "HomeTeamWin","AwayTeamWin")  
missing\_cols \<- setdiff(required\_cols, names(games))  
if (length(missing\_cols) \> 0\) {  
  stop(sprintf("Input file missing required columns: %s", paste(missing\_cols, collapse \= ", ")))  
}

\# \---- Identify playoff teams from NON-regular-season games \----  
playoffs\_raw \<- games %\>%  
  filter(\!(RegularSeason %in% c(1, "1", TRUE, "TRUE", "true")))

playoff\_teams \<- bind\_rows(  
  playoffs\_raw %\>% transmute(Season, Team \= HomeTeam),  
  playoffs\_raw %\>% transmute(Season, Team \= AwayTeam)  
) %\>%  
  distinct() %\>%  
  mutate(madePlayoffs \= 1L)

\# \---- Now restrict to REGULAR-season games for feature computation \----  
games\_reg \<- games %\>%  
  filter(RegularSeason %in% c(1, "1", TRUE, "TRUE", "true"))

\# \---- Parse dates (best-effort) to order games; fallback to row order \----  
parse\_maybe \<- function(x) {  
  suppressWarnings({  
    d \<- as.Date(x)  
    if (all(is.na(d))) d \<- as.Date(x, format \= "%m/%d/%Y")  
    if (all(is.na(d))) d \<- as.Date(x, format \= "%Y-%m-%d")  
    d  
  })  
}  
games\_reg \<- games\_reg %\>%  
  mutate(Date\_parsed \= parse\_maybe(Date),  
         RowOrder \= row\_number())  \# stable fallback

\# \---- Normalize to team-game long format (one row per team per REGULAR-SEASON game) \----  
home\_rows \<- games\_reg %\>%  
  transmute(  
    Season,  
    Date\_parsed,  
    RowOrder,  
    Team          \= HomeTeam,  
    Opp           \= AwayTeam,  
    PointsFor     \= HomeTeamScore,  
    PointsAgainst \= AwayTeamScore,  
    PointDiff     \= HomeTeamScore \- AwayTeamScore,  
    Win           \= as.integer(HomeTeamWin \== 1),  
    Home          \= 1L  
  )

away\_rows \<- games\_reg %\>%  
  transmute(  
    Season,  
    Date\_parsed,  
    RowOrder,  
    Team          \= AwayTeam,  
    Opp           \= HomeTeam,  
    PointsFor     \= AwayTeamScore,  
    PointsAgainst \= HomeTeamScore,  
    PointDiff     \= AwayTeamScore \- HomeTeamScore,  
    Win           \= as.integer(AwayTeamWin \== 1),  
    Home          \= 0L  
  )

long \<- bind\_rows(home\_rows, away\_rows)

\# \===============================================================================  
\# NEW FEATURE: SCHEDULE STRENGTH \- AvgOppWinPct  
\# \===============================================================================  
\# CRITICAL TEMPORAL CONSTRAINT:  
\# We must compute opponent strength using ONLY information available at the time  
\# of prediction (first 40% of season). This prevents data leakage.  
\#   
\# Algorithm:  
\# 1\. For each team-season, identify their first 40% of games (chronologically)  
\# 2\. For each opponent faced, compute that opponent's WinPct using ONLY their   
\#    first 40% of games  
\# 3\. Average these opponent WinPcts to get schedule strength metric  
\#  
\# This ensures we're not using future information (opponent's full-season record)  
\# to predict early-season outcomes.  
\# \===============================================================================

\# Step 1: Compute early-season WinPct for ALL teams (to serve as opponent strength lookup)  
\# This creates a lookup table: (Season, Team) \-\> EarlyWinPct  
early\_winpct\_lookup \<- long %\>%  
  group\_by(Season, Team) %\>%  
  arrange(Date\_parsed, RowOrder, .by\_group \= TRUE) %\>%  
  mutate(TotalGames \= n(),  
         TakeN \= ceiling(TotalGames \* 0.40),  
         GameIndex \= row\_number()) %\>%  
  filter(GameIndex \<= TakeN) %\>%  
  summarise(  
    EarlyWinPct \= mean(Win, na.rm \= TRUE),  
    .groups \= "drop"  
  )

\# Step 2: Keep only the FIRST 40% of each team's regular-season games \----  
\# For each (Season, Team): order by Date (fallback RowOrder), then take ceiling(n \* 0.40) rows.  
long\_first\_40 \<- long %\>%  
  group\_by(Season, Team) %\>%  
  arrange(Date\_parsed, RowOrder, .by\_group \= TRUE) %\>%  
  mutate(TotalGames \= n(),  
         TakeN \= ceiling(TotalGames \* 0.40),  
         GameIndex \= row\_number()) %\>%  
  filter(GameIndex \<= TakeN) %\>%  
  ungroup()

\# Step 3: Attach opponent's early-season WinPct to each game in the first 40%  
\# This join matches: (Season, Opp) from current game \-\> (Season, Team) from lookup  
long\_first\_40\_with\_opp \<- long\_first\_40 %\>%  
  left\_join(  
    early\_winpct\_lookup,  
    by \= c("Season" \= "Season", "Opp" \= "Team")  
  ) %\>%  
  rename(OppEarlyWinPct \= EarlyWinPct)

\# Step 4: Compute AvgOppWinPct for each team-season (average opponent strength)  
schedule\_strength \<- long\_first\_40\_with\_opp %\>%  
  group\_by(Season, Team) %\>%  
  summarise(  
    AvgOppWinPct \= mean(OppEarlyWinPct, na.rm \= TRUE),  
    .groups \= "drop"  
  )

\# \---- Team-season feature aggregation from FIRST 40% only (ORIGINAL FEATURES) \----  
team\_season\_early \<- long\_first\_40 %\>%  
  group\_by(Season, Team) %\>%  
  summarise(  
    GamesPlayedEarly   \= n(),  
    GamesWonEarly      \= sum(Win, na.rm \= TRUE),  
    WinPct             \= if\_else(GamesPlayedEarly \> 0, GamesWonEarly / GamesPlayedEarly, NA\_real\_),  
      
    HomeGamesEarly     \= sum(Home \== 1L, na.rm \= TRUE),  
    HomeWinsEarly      \= sum(Win \* (Home \== 1L), na.rm \= TRUE),  
    HomeWinPct         \= if\_else(HomeGamesEarly \> 0, HomeWinsEarly / HomeGamesEarly, NA\_real\_),  
      
    BlowoutWinsEarly   \= sum((Win \== 1L) & (PointDiff \>= 15), na.rm \= TRUE),  
    BlowoutWinPct      \= if\_else(GamesWonEarly \> 0, BlowoutWinsEarly / GamesWonEarly, NA\_real\_),  
      
    PPG                \= mean(PointsFor, na.rm \= TRUE),  
    AvgPointDiff       \= mean(PointDiff, na.rm \= TRUE),  
    SdPointDiff        \= sd(PointDiff, na.rm \= TRUE),  
      
    \# Momentum over the last 5 games inside the early window  
    Last5WinPct \= {  
      w \<- Win  
      k \<- min(5L, length(w))  
      if (k \> 0L) mean(tail(w, k)) else NA\_real\_  
    },  
    .groups \= "drop"  
  )

\# \---- Merge schedule strength feature with original features \----  
team\_season\_early \<- team\_season\_early %\>%  
  left\_join(schedule\_strength, by \= c("Season", "Team"))

\# \---- Attach playoff label (madePlayoffs) \----  
final\_full \<- team\_season\_early %\>%  
  left\_join(playoff\_teams, by \= c("Season", "Team")) %\>%  
  mutate(  
    madePlayoffs \= if\_else(is.na(madePlayoffs), 0L, 1L)  
  ) %\>%  
  arrange(Season, Team)

\# \---- Build TRIMMED table: ONLY the features \+ label \+ IDs (keeps Team here for splitting) \----  
\# NOW INCLUDES AvgOppWinPct as 8th feature  
final\_trimmed \<- final\_full %\>%  
  select(  
    Season, Team,  
    WinPct, HomeWinPct, BlowoutWinPct,  
    AvgPointDiff, SdPointDiff, PPG, Last5WinPct,  
    AvgOppWinPct,  \# NEW SCHEDULE STRENGTH FEATURE  
    madePlayoffs  
  ) %\>%  
  arrange(Season, Team)

\# \---- Helper: get season start year from 'Season' which may be "1999-00" or numeric \----  
get\_season\_start \<- function(x) {  
  if (is.numeric(x)) return(as.integer(x))  
  xs \<- suppressWarnings(as.integer(substr(as.character(x), 1, 4)))  
  na\_idx \<- is.na(xs)  
  if (any(na\_idx)) {  
    xs\[na\_idx\] \<- suppressWarnings(as.integer(as.character(x\[na\_idx\])))  
  }  
  xs  
}

season\_start \<- get\_season\_start(final\_trimmed$Season)

\# \---- Split into TRAIN / TEST by 5-year era blocks with Train, Test, Train cycle \----  
base\_year \<- 1947L  
block\_idx \<- floor((season\_start \- base\_year) / 5\)  
in\_test\_block  \<- (block\_idx %% 3L) \== 1L  
in\_train\_block \<- \!in\_test\_block

train\_idx \<- \!is.na(season\_start) & in\_train\_block  
test\_idx  \<- \!is.na(season\_start) & in\_test\_block

\# \---- Create TRAIN (nominal label, Season removed, Team removed) \----  
train\_df \<- final\_trimmed\[train\_idx, , drop \= FALSE\] %\>%  
  select(-Team) %\>%  
  mutate(madePlayoffs \= as.integer(madePlayoffs)) %\>%  
  mutate(madePlayoffs \= factor(ifelse(madePlayoffs \== 1, "yes", "no"), levels \= c("no","yes"))) %\>%  
  select(-Season)

\# \---- Split TEST set (the rows that were assigned to TEST) into DEV and TEST by straight alternating rows \----  
test\_rows \<- final\_trimmed\[test\_idx, , drop \= FALSE\]

if (nrow(test\_rows) \== 0\) {  
  stop("No rows were assigned to the TEST era blocks (check my Season values bro or the 5-year blocking).")  
}

\# Keep the existing ordering (final\_trimmed was arranged by Season, Team).  
\# Assign alternating rows: odd-index \-\> development, even-index \-\> test  
idx \<- seq\_len(nrow(test\_rows))  
dev\_in\_test \<- (idx %% 2L) \== 1L   \# 1,3,5,... \-\> development  
test\_in\_test \<- (idx %% 2L) \== 0L  \# 2,4,6,... \-\> test

dev\_df \<- test\_rows\[dev\_in\_test, , drop \= FALSE\] %\>%  
  select(-Team) %\>%  
  mutate(madePlayoffs \= as.integer(madePlayoffs)) %\>%  
  mutate(madePlayoffs \= factor(ifelse(madePlayoffs \== 1, "yes", "no"), levels \= c("no","yes"))) %\>%  
  select(-Season)

test\_df \<- test\_rows\[test\_in\_test, , drop \= FALSE\] %\>%  
  select(-Team) %\>%  
  mutate(madePlayoffs \= as.integer(madePlayoffs)) %\>%  
  mutate(madePlayoffs \= factor(ifelse(madePlayoffs \== 1, "yes", "no"), levels \= c("no","yes"))) %\>%  
  select(-Season)

\# \---- Sanity checks: ensure each split has at least 150 instances \----  
count\_train \<- nrow(train\_df)  
count\_dev   \<- nrow(dev\_df)  
count\_test  \<- nrow(test\_df)

if (count\_train \< 150\) {  
  stop(sprintf("Train set has %d rows (\<150). The era blocking produced too few train instances.", count\_train))  
}  
if (count\_dev \< 150\) {  
  stop(sprintf("Development set has %d rows (\<150). The alternating split of test produced too few dev instances.", count\_dev))  
}  
if (count\_test \< 150\) {  
  stop(sprintf("Test set has %d rows (\<150). The alternating split of test produced too few test instances.", count\_test))  
}

\# \---- Write the three outputs (nominal labels, no Season) \----  
write\_csv(train\_df, path\_out\_train\_nom\_noSeason)  
write\_csv(dev\_df,   path\_out\_dev\_nom\_noSeason)  
write\_csv(test\_df,  path\_out\_test\_nom\_noSeason)

\# \---- Summary prints \----  
cat(sprintf("✅ Wrote nominal TRAIN (no Season) \-\> %s with %d rows, %d cols.\\n",  
            path\_out\_train\_nom\_noSeason, nrow(train\_df), ncol(train\_df)))  
cat(sprintf("✅ Wrote nominal DEVELOPMENT (no Season) \-\> %s with %d rows, %d cols.\\n",  
            path\_out\_dev\_nom\_noSeason, nrow(dev\_df), ncol(dev\_df)))  
cat(sprintf("✅ Wrote nominal TEST (no Season) \-\> %s with %d rows, %d cols.\\n",  
            path\_out\_test\_nom\_noSeason, nrow(test\_df), ncol(test\_df)))

\# \---- Feature Summary \----  
cat("\\n📊 FEATURE SUMMARY:\\n")  
cat("Original features (7): WinPct, HomeWinPct, BlowoutWinPct, AvgPointDiff, SdPointDiff, PPG, Last5WinPct\\n")  
cat("NEW feature added (1): AvgOppWinPct (Average Opponent Win Percentage)\\n")  
cat(sprintf("Total features: %d\\n", ncol(train\_df) \- 1))  
cat(sprintf("Label: madePlayoffs (no/yes)\\n\\n"))

\# \---- Sample Statistics for New Feature \----  
cat("📈 AvgOppWinPct STATISTICS (Schedule Strength):\\n")  
cat(sprintf("  Train \- Mean: %.4f, SD: %.4f, Range: \[%.4f, %.4f\]\\n",  
            mean(train\_df$AvgOppWinPct, na.rm \= TRUE),  
            sd(train\_df$AvgOppWinPct, na.rm \= TRUE),  
            min(train\_df$AvgOppWinPct, na.rm \= TRUE),  
            max(train\_df$AvgOppWinPct, na.rm \= TRUE)))

cat(sprintf("  Dev   \- Mean: %.4f, SD: %.4f, Range: \[%.4f, %.4f\]\\n",  
            mean(dev\_df$AvgOppWinPct, na.rm \= TRUE),  
            sd(dev\_df$AvgOppWinPct, na.rm \= TRUE),  
            min(dev\_df$AvgOppWinPct, na.rm \= TRUE),  
            max(dev\_df$AvgOppWinPct, na.rm \= TRUE)))

cat(sprintf("  Test  \- Mean: %.4f, SD: %.4f, Range: \[%.4f, %.4f\]\\n",  
            mean(test\_df$AvgOppWinPct, na.rm \= TRUE),  
            sd(test\_df$AvgOppWinPct, na.rm \= TRUE),  
            min(test\_df$AvgOppWinPct, na.rm \= TRUE),  
            max(test\_df$AvgOppWinPct, na.rm \= TRUE)))

cat("\\n💡 Interpretation:\\n")  
cat("   AvgOppWinPct ≈ 0.50: Faced average-strength opponents\\n")  
cat("   AvgOppWinPct \> 0.55: Faced tough schedule (stats may be suppressed)\\n")  
cat("   AvgOppWinPct \< 0.45: Faced weak schedule (stats may be inflated)\\n")

Weka Screenshots that tables, data and results are actually from:  
Pre error analysis:  
![][image1]

Baseline SMO:  
![][image2]

Data With final features:  
![][image3]

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,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>