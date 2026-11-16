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
