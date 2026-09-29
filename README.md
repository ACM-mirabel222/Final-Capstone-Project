# Final-Capstone-Project
Understanding and Predicting Father Absence in South Africa Using Machine Learning for Social Insight and Policy Support.
A machine learning framework and interactive app that predicts whether a South African individual lives with their father, built on national household survey data to support social policy and early intervention.

Capstone project · Sol Plaatje University · Supervisor: Dr. Ibidun Obagbuwa

#### Table of contents
1. Overview
2. Key findings
3. Screenshots
4. Data
5. Methodology
6. Model results
7. Repository structure
8. Getting started
9. Using the app
10. Limitations and ethical notes
11. Author and acknowledgements
12. References

#### Overview
Fatherlessness is a widely discussed social issue in South Africa, yet little empirical work uses machine learning to identify which factors are linked to a father being absent from the household. This project fills that gap.

Using the national General Household Survey, it:
- explores demographic, socio-economic, health and educational indicators,
- trains supervised classifiers to predict father presence (1 = present, 0 = absent),
- surfaces the strongest predictors in an interpretable way, and
- delivers the results through an interactive dashboard and prediction web app.

Intended users: social workers, educators, NGOs, government agencies and policymakers who need to target support at higher-risk households.

#### Key findings
- 69% of individuals in the analysed sample do not live with their father, and 31% do. (This is limited to people whose father is known to be alive.)
- Household structure is the strongest signal. hhc_relationship and hhc_moth_parthh (whether the biological mother is in the household) rank highest in the Random Forest importance scores.
- Socio-economic variables such as social grants, education, province and age group also carry predictive power.
- Father absence is highest among children aged 0–14, and varies by province and settlement type (urban, traditional, farms).
- XGBoost performed best overall and is the recommended model for deployment. Logistic regression is useful for interpretability, and Random Forest is a robust alternative.

| These are associations in survey data, not proof of cause. See Limitations. |

#### Screenshots
Dashboard	& Prediction app
<img width="616" height="309" alt="Screenshot 2026-06-17 162837" src="https://github.com/user-attachments/assets/c6aaf0be-e35e-4f06-94df-22dc99657c3a" />

An interactive Tableau dashboard is also available: Tableau Public link

#### Data
Item	           -  Detail
Source	         -  General Household Survey (GHS), Statistics South Africa, distributed via DataFirst, University of Cape Town
Files used	     -  ghs-2024-person-v1.csv (analysis data) and ghs-2023-person-v1.csv (code/label reference)
Size	           -  70,440 rows × 114 columns (2024 person file)
Modelling subset -  36,883 rows where the father is known to be alive and his household status is known (25,453 "No", 11,430 "Yes")
Target variable	 -  hhc_fath_parthh: is the biological father part of the household?

The raw data is not included in this repository. DataFirst's terms of use apply, so please download the files yourself:
1. Register and request access at DataFirst.
2. Download the 2023 and 2024 GHS person files.
3. Place them in data/raw/ (this folder is git-ignored).

#### Methodology
The project follows a standard data science pipeline, with a focus on interpretability.
1. Data acquisition: GHS person files from DataFirst.
2. Preprocessing: the dataset had no missing values or duplicates. The main challenge was that the 2024 file stores categories as text labels while the 2023 file uses coded labels (Code. Label). Labels were mapped to the 2023 codes (with manual mapping from the survey guide for anything unmatched), and then only the numeric codes were kept so the data is ML-ready.
3. Filtering: keep records where the father is alive and his household status is "Yes" or "No"; recode the target to 1/0.
4. Exploratory analysis: correlation heatmap and distributions by age group, province, race, and settlement type, plus Tableau bubble charts.
5. Handling imbalance: random under-sampling to balance the classes before training.
6. Modelling (in R): Random Forest, XGBoost and binomial logistic regression, evaluated with accuracy, balanced accuracy, Cohen's Kappa, AUC and OOB error.
7. Deployment: an R Plumber API serves predictions; a Python Panel app provides the user interface, dashboard and feedback loop.

Tools: Python (pandas, numpy, Plotly, Panel, seaborn), R (Plumber and model packages), Tableau.

#### Model results
Evaluated on the balanced dataset.

Model	                  Accuracy	Balanced accuracy	Kappa	AUC
XGBoost (preferred)	    0.815	    0.815	   0.630	  0.876
Random Forest	          0.797	    0.761	   0.523	  0.856
Logistic regression   	~0.79	    n/a      n/a	    0.816

Top predictors (Random Forest, MeanDecreaseGini): household relationship, mother's presence in household, province, age group, population group, marital status, social grants.

#### Repository structure
.
├── README.md
├── LICENSE
├── requirements.txt              # Python dependencies
├── data/
│   └── raw/                      # GHS CSVs go here (not committed)
├── notebooks/
│   └── father_presence_analysis.ipynb   # cleaning, EDA, dashboard, app code
├── r/
│   ├── train_models.R            # Random Forest, XGBoost, logistic regression
│   └── plumber.R                 # API: /predict, /feedback, /retrain
├── app/
│   ├── app.py                    # Panel app
│   └── assets/                   # logo.png, cover image
├── docs/
│   ├── images/                   # screenshots for this README
│   └── Capstone_Project_final_report.pptx
└── .gitignore

#### Getting started
##### Prerequisites
- Python 3.10+ and R 4.x
- The GHS data files (see Data)

##### 1. Clone the repository
bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

##### 2. Python environment
bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

Main packages: pandas, numpy, matplotlib, seaborn, plotly, panel, requests, folium, geopy.

##### 3. R packages
r
install.packages(c("plumber", "randomForest", "xgboost", "caret", "pROC"))

##### 4. Prepare the data
Update the data folder path at the top of the notebook (it currently points to a Google Drive folder) so it reads from data/raw/, then run the cleaning cells to produce the modelling dataset.

##### 5. Start the prediction API
r
plumber::pr("r/plumber.R") |> plumber::pr_run(port = 47260)

The app expects the API at http://127.0.0.1:47260. Change API_URL in the app if you use a different port.

##### 6. Launch the app
bash
panel serve app/app.py --show

#### Using the app
Page	          - What it does
Home	          - Project introduction
Prediction	    - Enter values by category (education, labour, social grants, household, demographics, employment, health) and click Predict to get the predicted class and probabilities
Dashboard	      - Interactive charts: correlation heatmap, age group, province, race, settlement type, overall split, plus a link to the Tableau dashboard
Findings	      - Written summary of the analysis
Recommendations	- Suggested policy actions

##### API endpoints
Endpoint	     - Purpose
POST /predict	 - Returns the predicted class and probability_yes / probability_no
POST /feedback - Adds a user-confirmed row to the training data
POST /retrain	 - Retrains the model and reports the OOB error

#### Limitations and ethical notes
- Correlation, not causation. The model and charts show associations. For example, social grant receipt likely reflects poverty and child-support payments to caregivers, not a cause of father absence.
- Presence in the household is not the same as involvement. A father can live elsewhere and still be actively involved, or live at home and be uninvolved.
- Scope. Deceased or unknown fathers and longitudinal tracking are out of scope. Results describe the survey sample, and counts may not match national totals unless survey weights are applied.
- Class imbalance. Under-sampling improves balance but discards data; performance on the majority and minority classes differs (sensitivity is lower for "father absent").
- Sensitive data. The survey is anonymised, and the models use coded categories. Do not commit raw data or any user-submitted feedback rows that could identify a household.
- Responsible use. Predictions are meant to help target support, not to label or penalise individuals or families.

#### Author and acknowledgements
Asnath Miandabu, Sol Plaatje University Supervisor: Dr. Ibidun Obagbuwa

Data provided by Statistics South Africa via DataFirst (University of Cape Town).

Contact: www.linkedin.com/in/asnath-miandabu-2b6522211

#### References
- DataFirst UCT. (2023). General Household Survey 2023. https://www.datafirst.uct.ac.za/services/citations
- News24. (2025). New study highlights the impact of fatherlessness in South Africa. https://www.news24.com
- Daily Maverick. (2023). South African society suffering profound absence of father. https://www.dailymaverick.co.za/opinionista/2023-06-14-its-fathers-day-but-south-african-society-suffering-profound-absence-of-father-figures/
- Mavungu Eddy, M., Thomson-de Boor, H. and Mphaka, K. (2013). "So we are ATM fathers": A study of absent fathers in Johannesburg, South Africa. Centre for Social Development in Africa, University of Johannesburg.
