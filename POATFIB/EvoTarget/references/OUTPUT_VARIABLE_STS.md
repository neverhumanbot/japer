# STS CABG Postoperative & Intraoperative Variables (Outputs / Leakage)

These variables represent outcomes that occur **during or after** the surgery, as well as STS calculated downstream risk equivalents. Using these as inputs for preoperative predictions results in temporal data leakage.

## 1. Postoperative Outcome Targets (Strict Outputs)
* `COtAFib` - Postoperative Atrial Fibrillation (Main AF outcome target)
* `Mortalty` - Operative or post-operative Mortality

## 2. STS Algorithmic Risk Scores (Outputs/Leakage Proxies)
These variables are the calculated risk estimates provided by the STS calculator using preoperative data. They should **not** be used as inputs for building new models.
* `PredMort` - STS Predicted Mortality Risk
* `PredMM` - STS Predicted Morbidity/Mortality Risk
* `PredStro` - STS Predicted Stroke Risk
* `PredRenF` - STS Predicted Renal Failure Risk
* `PredVent` - STS Predicted Prolonged Ventilation Risk
* `Pred6D` - STS Predicted Risk of a 6-day stay
* `Pred14D` - STS Predicted Risk of a 14-day stay
* `PredDeep` - STS Predicted Deep Sternal Wound Infection Risk

## 3. Postoperative Complications & Interventions
* `CSepsis` - Postoperative Sepsis
* `CPVntLng` - Postoperative Prolonged Ventilation
* `CPPneum` - Postoperative Pneumonia
* `CVTE` - Postoperative Venous Thromboembolism
* `CRenFail` - Postoperative Renal Failure
* `MtOpD` - Mortality operative details

## 4. Postoperative Labs & Recovery Status
* `PostCreat` - Postoperative Creatinine level
* `PostOpPeakGlu` - Highest Postoperative Peak Glucose level
* `VentHrsA` - Postoperative Ventilation Duration (hours)
* `ICUInHrs` - Total ICU Duration (hours)
* `LOS` - Total Hospital Length of Stay
* `PLOS` - Postoperative Length of Stay

## 5. Discharge Medications
* `DCASA` - Aspirin prescribed at discharge
* `DCBeta` - Beta blocker prescribed at discharge
* `DCAmiodarone` - Amiodarone prescribed at discharge

## 6. Intraoperative Variables (Intra-Surgery Leakage)
These occur during the actual procedure and cannot properly be viewed as "Preoperative" predictors.
* `CPBUtil` - Cardiopulmonary bypass utilized
* `XClampTm` - Cross-clamp time
* `PerfusTm` - Total bypass perfusion time
* `LwstTemp` - Lowest intraoperative core temperature
* `HighIntraGlu` - Highest intraoperative glucose level
* `CABDistSite01` (and related) - Coronary locations bypassed
* `CABConduit` - Types of conduits used for bypass grafts
* `IMedEACA` - Intraoperative medication given (e.g., Epsilon-aminocaproic acid)
