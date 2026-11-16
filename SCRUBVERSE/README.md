# SCRUBVERSE – Data Rescue Mission

**Course:** Introduction to Data Science for Engineers  
**Course Code:** CS3006  
**Project:** Interactive Data Cleaning Learning Game

## Project Overview

SCRUBVERSE – Data Rescue Mission is a browser-based educational game that introduces important data-cleaning concepts through a nine-level mission. Players act as data rescuers: they investigate a corrupted dataset, identify data-quality problems, choose suitable cleaning methods, and review the results.

The game combines a visual dataset interface, short problem-solving tasks, Pandas code-completion questions, hints, scoring, and explanations. It is designed to make data cleaning more interactive and help learners understand not only which method to use, but also why it is appropriate.

## Team Members

| Name | Roll Number |
|---|---|
| Gayathri | EC24I2024 |
| Snehaja | EC24I1015 |
| Nishitha | EC24I1002 |

## Learning Objectives

Through the mission, players practise how to:

- Detect and count missing values.
- Decide when dropping incomplete rows is appropriate.
- Impute missing values using mean, median, or mode.
- Identify and remove duplicate records.
- Distinguish genuine extreme values from erroneous outliers.
- Standardise inconsistent categorical labels.
- Identify and correct incorrect data types.
- Audit a dataset and apply a sequence of cleaning operations.
- Validate the cleaned data and understand the impact of cleaning decisions.

## The Nine Mission Levels

| Level | Mission | Main concept |
|---|---|---|
| 1 | Spot the Missing | Detect and count missing values using `isnull()` / `isna()` |
| 2 | Drop It | Identify missing-data columns and understand `dropna()` |
| 3 | Impute It | Choose mean, median, or mode according to the column |
| 4 | Duplicate Hunt | Count and remove exact duplicate rows |
| 5 | Outlier Check | Separate invalid values from genuine extreme observations |
| 6 | Messy Categories | Clean whitespace and standardise category labels |
| 7 | Type Fixer | Identify and convert incorrect data types |
| 8 | Trap Level | Understand why dropping rows with any missing value can cause major data loss |
| 9 | Boss Level | Perform a combined data-quality audit and clean multiple issue types |

## How the Game Works

1. **Start the mission:** The welcome screen introduces the data-rescue scenario and loads the bundled sample dataset. Players can alternatively select their own CSV file.
2. **Explore the mission map:** The map displays the nine levels. Levels are unlocked progressively as the player completes the previous mission.
3. **Investigate the data:** Each level presents a dataset view and questions about the issue being investigated. Players can inspect columns and cells and switch between the original view and the working copy.
4. **Identify the issue:** Players answer counting, selection, classification, or Pandas code-completion questions. Incorrect answers lead to an Emergency Room screen, where hints from different in-game experts are available.
5. **Clean the data:** In levels with a cleaning phase, players use the available actions—such as dropping rows, filling missing values, removing duplicates, correcting selected values, standardising text, or converting data types—within an action-point budget.
6. **Review the result:** The Appreciation Room displays the score breakdown, before-and-after information, an explanation of the correct approach, alternative methods, Pandas syntax, and a real-world application.
7. **Revise and finish:** Completed-level notes are collected in the Data Scientist's Notebook. After all nine levels, the Mission Results screen shows the level scores, stars, total score, and a browser-local leaderboard.

## Dataset and Data Handling

The bundled sample is an occupancy and environmental-sensor dataset with columns including:

- `date`
- `Temperature`
- `Humidity`
- `Light`
- `CO2`
- `HumidityRatio`
- `Occupancy`

The game also introduces simulated data-quality issues for particular missions, such as missing values, duplicate rows, outliers, inconsistent category labels, and incorrect data types. These simulated cases let players practise cleaning decisions in a controlled learning environment.

Players may upload a CSV file through the welcome screen. The game states that the original CSV is not modified; cleaning operations are performed on a separate working copy in the browser.

## Related Data Science Case Study: Early Heart Stroke Prediction

The team's **Early Heart Stroke Prediction** report provides a real-world healthcare example of why careful data understanding, cleaning, preprocessing, and model evaluation matter. This case study is documented here as a related data-science project; it is separate from the occupancy-sensor sample bundled inside the SCRUBVERSE game.

### Problem and objective

The case study uses patient demographic, lifestyle, and medical attributes to estimate whether a patient is likely to have a stroke. It is a binary classification task, where `0` represents no stroke and `1` represents stroke. The objective was to build an early risk-screening model and document the complete data-science pipeline.

### Dataset

The report uses a public secondary healthcare stroke dataset sourced from Kaggle. The received raw file contained 6,312 rows and 12 columns:

- `id`, `gender`, `age`
- `hypertension`, `heart_disease`
- `ever_married`, `work_type`, `Residence_type`
- `avg_glucose_level`, `bmi`, `smoking_status`
- `stroke` (target)

The raw file contained 413 missing values in `bmi` and 1,202 exact duplicate rows when the `id` field was ignored. The duplicate records distorted the apparent stroke proportion: the raw count suggested approximately 23%, whereas the cleaned dataset had a stroke rate of 4.87% (249 cases among 5,109 rows).

### Data preparation and modelling workflow

The report follows a nine-stage data-science pipeline:

1. Define the prediction problem and choose recall as the primary success metric.
2. Load the Kaggle CSV and inspect its dimensions and sample records.
3. Examine data types, summary statistics, and target-class distribution.
4. Clean the data by handling missing values, duplicate rows, the identifier column, a rare gender category, and outliers.
5. Perform exploratory data analysis to examine distributions, correlations, and possible risk-factor patterns.
6. Preprocess the data using one-hot encoding, a stratified 80/20 train-test split, training-set median imputation for BMI, and feature scaling fitted on training data.
7. Train Logistic Regression and Random Forest classifiers with balanced class weights.
8. Evaluate the models using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, train-versus-test comparison, and 5-fold stratified cross-validation.
9. Save the selected model and provide a prediction function for new patient records.

### Key findings reported

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.766 | 0.838 |
| Precision | 0.143 | 0.174 |
| Recall | 0.760 | 0.620 |
| F1-score | 0.241 | 0.272 |
| ROC-AUC | 0.850 | 0.829 |

Logistic Regression was selected as the better screening model because it detected 38 of the 50 stroke cases in the test set (recall 0.76), compared with 31 for Random Forest (recall 0.62). Its mean cross-validation recall was 0.804 ± 0.083. Random Forest showed signs of overfitting, with training recall of 1.00 and test recall of 0.62.

The report notes that precision was low (approximately 0.14–0.17), meaning that many patients flagged by the model were false alarms. Therefore, the prediction is intended only to support screening and prioritisation for medical review; it is not a diagnosis and must not replace a clinician's judgement.

### Connection to the learning goals

This case study demonstrates why the data-cleaning concepts practised in SCRUBVERSE matter in a real analysis:

- Duplicate detection can prevent misleading class proportions and data leakage.
- Missing-value imputation should be fitted on training data to avoid leakage from the test set.
- Outliers need to be investigated rather than automatically deleted; the report retained extreme BMI and glucose values because they were medically possible.
- Class imbalance makes accuracy alone an inadequate measure; recall, precision, F1-score, and ROC-AUC provide additional insight.
- Comparing training, test, and cross-validation results helps reveal overfitting and assess generalisation.

**Important distinction:** SCRUBVERSE teaches data-cleaning concepts through its browser-based missions. The stroke-prediction model and its reported metrics belong to the separate Python/scikit-learn case study in the team's report; the game does not run that machine-learning model.

## Technologies Used

- **HTML5** – page structure and game screens
- **CSS3** – responsive layout, visual styling, animations, and game interface
- **JavaScript** – dataset handling, mission flow, validation, scoring, and interaction
- **Pandas concepts** – represented through explanatory examples and code-completion questions

The project is a self-contained front-end web game. Pandas syntax is taught through the in-game examples; the browser game itself does not execute Python or run the Pandas library.

## How to Run

### Run locally

1. Download or clone this repository.
2. Open `index.html` in a modern web browser.
3. Select **START MISSION**.
4. Follow the levels in order. You can use the bundled dataset or upload a CSV file.

No package installation or build step is required.

### Publish with GitHub Pages

1. Open the repository on GitHub.
2. Go to **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select the `main` branch and the `/ (root)` folder, then save.
5. Once GitHub Pages finishes deploying, the published site URL will be shown in the Pages settings.

## Important Notes

- The original uploaded CSV is kept separate from the in-game working copy.
- Mission issues may be simulated to support specific learning objectives.
- Scores and the leaderboard are stored in the browser's local storage, so they are local to that browser and device.
- The game is intended as a learning activity and demonstration of data-cleaning concepts.

---

**SCRUBVERSE – Data Rescue Mission**  
*Investigate the data. Clean the mess. Rescue the dataset.*
