# STS CABG Preoperative Variables (Safe Inputs)

These variables are defined as "preoperative" and are safe to use as strict inputs for predictive modeling (e.g., predicting post-operative outcomes like Atrial Fibrillation or Mortality risk) without risking temporal data leakage.

### Demographics & Anthropometrics
* `Age` / `Age1` - Patient age (in years)
* `Gender` - Biological sex (1=Male, 2=Female)
* `BSA` - Body Surface Area (m²)
* `HeightCm` - Patient height in centimeters
* `WeightKg` - Patient weight in kilograms
* `Race*` - Race flags (binary indicators for various races)
* `Ethnicity` - Hispanic/Non-Hispanic status

### Major Comorbidities (Diagnosed before surgery)
* `Diabetes` - History of Diabetes Mellitus (1=Yes, 2=No, 3=Unknown)
* `CHF` - Congestive Heart Failure history (1=Yes, 2=No, 3=Unknown)
* `Dialysis` - Preoperative dialysis requirement (1=Yes, 2=No, 3=Unknown)
* `PVD` - Peripheral Vascular Disease indicating atherosclerotic burden (1=Yes, 2=No, 3=Unknown)
* `CVA` - Cerebrovascular Accident (prior stroke)
* `CVD` - Cerebrovascular Disease history
* `Hypertn` - Hypertension diagnosis
* `Dyslip` - Dyslipidemia diagnosis
* `Cancer` - History of cancer
* `PrevMI` - Previous Myocardial Infarction
* `Arrhythmia` - History of arrhythmia (generic)
* `ArrhythAFib` - Preoperative Atrial Fibrillation history

### Preoperative Lab Values & Biomarkers
* `CreatLst` - Last preoperative creatinine level (Baseline renal function)
* `Hct` - Baseline Hematocrit
* `WBC` - White Blood Cell count
* `Platelets` - Platelet count
* `BNP` - B-type Natriuretic Peptide
* `NTproBNP` - N-terminal pro b-type natriuretic peptide
* `hsTnT` - High-sensitivity Troponin T

### Pulmonary & Respiratory
* `PFT` - Pulmonary Function Test results
* `FEV1` - Forced Expiratory Volume in 1 second
* `DLCOPred` - Predicted DLCO (Diffusing capacity of the lungs for carbon monoxide)

### Preoperative Interventions & Anatomy
* `PrCAB` - Prior Coronary Artery Bypass Graft surgery
* `PrValve` - Prior Valve surgery
* `POCPCI` - Prior Percutaneous Coronary Intervention
* `NumDisV` - Number of diseased coronary vessels
* `HDEF` - Ejection Fraction indicator
* `PASYS` - Pulmonary Artery Systolic Pressure

### Preoperative Medications (Taken prior to admission/surgery)
* `MedBeta` - Preoperative use of Beta blockers
* `MedAmiodarone` - Preoperative use of Amiodarone
* `MedASA` - Preoperative use of Aspirin

### Surgical Status Protocol
* `Status` - Preoperative Surgical Urgency (e.g., 1=Elective, 2=Urgent)
