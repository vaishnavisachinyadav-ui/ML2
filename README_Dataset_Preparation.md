# 🧹 Prepare the Titanic Dataset for Machine Learning

An end-to-end Data Preparation project demonstrating data inspection, missing-value handling, exploratory data analysis (EDA), encoding and scaling to turn the raw **Titanic Passenger Dataset** into a clean, machine-learning-ready dataset using **Pandas** and **Scikit-Learn**.

**Activity 1 | Unit I – Foundations & Data Preparation | B.Tech – AI & Data Science**

---

## 📋 Overview

This project prepares the Titanic Passenger Dataset for a future machine learning workflow. It inspects the data structure, identifies data-quality issues (missing values, duplicates, repeated columns, mixed data types), performs basic EDA with visualisations, encodes categorical features, scales numerical features and saves the final prepared dataset as a CSV file. **No prediction model is built**, as required by the activity.

---

## 📊 Dataset: Titanic Passengers

The dataset is loaded directly from the Seaborn data repository: [`titanic.csv`](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv). It contains **891 records** with **14 passenger attributes** and **1 possible target variable** (`survived`). Each row represents one passenger.

### Features

| Feature | Description |
| :--- | :--- |
| **`pclass`** | Ticket class (1 = First, 2 = Second, 3 = Third) |
| **`sex`** | Sex of the passenger (male / female) |
| **`age`** | Age in years |
| **`sibsp`** | Number of siblings / spouses aboard |
| **`parch`** | Number of parents / children aboard |
| **`fare`** | Ticket price |
| **`embarked`** | Port of boarding (S = Southampton, C = Cherbourg, Q = Queenstown) |
| **`class`** | Ticket class as text (repeats `pclass`) |
| **`who`** | Person type (man / woman / child) |
| **`adult_male`** | True if the passenger is an adult male |
| **`deck`** | Deck of the cabin |
| **`embark_town`** | Port of boarding as text (repeats `embarked`) |
| **`alive`** | Survival as text (repeats `survived`) |
| **`alone`** | True if the passenger travelled alone |

### Target Variable

* **`survived`**: Survival indicator (0 = did not survive, 1 = survived). Identified as the possible target for a future classification task.

---

## 🧠 Methodology: Data Preparation Techniques

### Missing-Value Imputation

* **Median** for skewed numerical columns (`age`): the median is not affected by extreme values.
* **Mode** for categorical columns (`embarked`, `embark_town`): the most frequent category.
* **Column removal** when most of the data is missing (`deck`, 77% missing).

### One-Hot Encoding

Nominal categories with no natural order are converted into 0/1 columns. With `drop_first=True`, a category with $k$ values creates $k-1$ columns to avoid repeated information.

### Standardisation

Numerical features are rescaled so that they have mean 0 and standard deviation 1:

$$z = \frac{x - \mu}{\sigma}$$

Where:
* $x$ is the original value
* $\mu$ is the mean of the **training data**
* $\sigma$ is the standard deviation of the **training data**
* $z$ is the scaled value

The scaling parameters are learned from the training rows only (80% split) and then applied to all rows, which avoids information leaking from the test data.

---

## 📈 Project Pipeline

The workflow in [`Dataset_Preparation_Activity_1.ipynb`](./Dataset_Preparation_Activity_1.ipynb) follows these key steps:

1. **Dataset Selection**: Recording the dataset name, source link, file type, rows and columns.
2. **Load & Inspect**: Displaying the first and last 5 rows, shape, column names and data types.
3. **Data Types & Statistics**: Separating numerical and categorical columns, reviewing `df.describe()`, identifying the target and noting suspicious values.
4. **Missing-Value Detection**: Counting missing values and percentages for every column, with a bar chart.
5. **Missing-Value Handling**: Working on a copy of the data and applying a suitable method to each column.
6. **Duplicates & Consistency**: Inspecting duplicate rows, checking category labels and removing repeated columns.
7. **Basic EDA**: Summary statistics, frequency counts and four visualisations.
8. **Encoding**: One-hot encoding of nominal columns with a before-and-after comparison.
9. **Scaling**: Standardising `age`, `sibsp`, `parch` and `fare` using training data only.
10. **Save Prepared Data**: Final checks, saving to CSV and writing the conclusion.

---

## 📊 Preparation Results

### Data Quality Findings

| Issue | Finding | Decision |
| :--- | :--- | :--- |
| **`deck`** | 688 missing values (**77.22%**) | **Column dropped**, too much missing to fill reliably |
| **`age`** | 177 missing values (**19.87%**) | Filled with the **median** (`28.0`) |
| **`embarked`** | 2 missing values (**0.22%**) | Filled with the **mode** (`S`) |
| **`embark_town`** | 2 missing values (**0.22%**) | Filled with the **mode** (`Southampton`) |
| **Duplicate rows** | **107** in the original data (116 after filling ages) | **Kept**, no passenger ID exists, so they are most likely different passengers |
| **Repeated columns** | `class`, `alive`, `embark_town` repeat other columns | **Dropped** |
| **Suspicious values** | 15 passengers with `fare = 0`; maximum fare `512.33` | **Kept**, unusual but possible |
| **Label consistency** | No spelling differences or extra spaces | No correction needed |

### EDA Observations

| Observation | Result |
| :--- | :--- |
| **Survival by sex** | **74.2%** of women survived vs only **18.9%** of men |
| **Survival by class** | **63.0%** (First) → **47.3%** (Second) → **24.2%** (Third) |
| **Fare by class** | Median fare about `60` (First) vs about `8` (Third), with a few very expensive outliers |
| **Age** | Most passengers are young adults; the spike near 28 comes from the median fill |

### Encoding & Scaling

| Columns | Technique | Reason |
| :--- | :--- | :--- |
| **`sex`, `embarked`, `who`** | One-hot encoding | Nominal categories with no natural order |
| **`pclass`** | Kept as ordered number | The order (1 < 2 < 3) is meaningful |
| **`adult_male`, `alone`** | Converted from True/False to 1/0 | Numeric form required |
| **`age`, `sibsp`, `parch`, `fare`** | Standardisation (`StandardScaler`) | Features have very different ranges (for example `fare` up to 512, `parch` up to 6) |

### Final Prepared Dataset

* **Shape:** `891 rows × 13 columns`
* **Missing values:** `0`
* **All columns numeric**, ready for a future ML task (target: `survived`)
* **Columns:** `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`, `adult_male`, `alone`, `sex_male`, `embarked_Q`, `embarked_S`, `who_man`, `who_woman`

### Output Files

| File | Description |
| :--- | :--- |
| **`titanic_original.csv`** | Original dataset, unchanged (891 × 15) |
| **`titanic_cleaned.csv`** | After cleaning, before encoding and scaling (891 × 11) |
| **`titanic_prepared.csv`** | **Final prepared dataset** (891 × 13) |

---

## 🛠️ Tech Stack

* **Language**: Python 3.x
* **Data Manipulation**: `pandas`, `numpy`
* **Visualization**: `matplotlib`
* **Preprocessing**: `scikit-learn` (`StandardScaler`, `train_test_split`)
* **Development Environment**: Google Colab / Jupyter Notebook (`.ipynb`)

---

## 📦 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/<your-username>/Titanic-Dataset-Preparation.git
cd Titanic-Dataset-Preparation
pip install pandas numpy matplotlib scikit-learn notebook
```

### 2. Run the Notebook

Launch Jupyter Notebook to view and run the analysis:

```bash
jupyter notebook Dataset_Preparation_Activity_1.ipynb
```

Or upload the notebook to [Google Colab](https://colab.research.google.com) and click **Runtime → Run all**. (Internet is needed to load the dataset.)

### 3. Quick Code Example

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load Titanic Dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

# 2. Handle Missing Values
df = df.drop(columns=['deck'])                                     # 77% missing
df['age'] = df['age'].fillna(df['age'].median())                   # median
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])   # mode

# 3. Drop Repeated Columns & Convert Booleans
df = df.drop(columns=['class', 'alive', 'embark_town'])
df[['adult_male', 'alone']] = df[['adult_male', 'alone']].astype(int)

# 4. One-Hot Encode Nominal Columns
df = pd.get_dummies(df, columns=['sex', 'embarked', 'who'], drop_first=True, dtype=int)

# 5. Scale Numerical Features (fit on training rows only)
scale_cols = ['age', 'sibsp', 'parch', 'fare']
train_idx, _ = train_test_split(df.index, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(df.loc[train_idx, scale_cols])
df[scale_cols] = scaler.transform(df[scale_cols])

# 6. Save Prepared Dataset
df.to_csv("titanic_prepared.csv", index=False)
print(df.shape, "missing:", df.isnull().sum().sum())
# Output: (891, 13) missing: 0
```

---

## ✅ Conclusion

The raw Titanic data had a heavily incomplete column (`deck`), a moderately incomplete column (`age`), repeated columns, text and True/False columns, and features on different scales. After preparation, the final dataset has no missing values, all numeric columns and standardised features, and is ready for machine learning.

**Key EDA insight:** Women and First-class passengers had much higher survival rates than men and Third-class passengers.

---

## 📄 License

This project is created for academic and educational purposes. The Titanic dataset is publicly available from the [Seaborn data repository](https://github.com/mwaskom/seaborn-data).
