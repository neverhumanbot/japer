# Export fields with documented pre-incision timing

- `Age1` is patient age in years. See Demographics & Anthropometrics in INPUT_VARIABLE_STS.md and STS v2.81 Seq 70 for age at the date of surgery.
- `BSA` is body surface area from height and weight. See Demographics & Anthropometrics in INPUT_VARIABLE_STS.md and the BSA entry in STS_CABG_Key_Variables_Full_Detailed_Report.md.
- `Region` is the coarse geography already known before surgery. The archived CT-MGN preoperative extractor (recovered in Gemini handoff step 49) includes it as context and lists geographic codes such as PACIFIC, MIDATL and GRLAKES. It is kept as a category.
- `CathtoSurg` is elapsed days from preoperative catheterization to surgery. The two event dates are documented in STS v2.81 Seq 1150 (cardiac catheterization date) and Seq 310 (surgery date). At the prediction cutoff the interval is already known. Negative intervals are invalid for this interpretation.
- `CountryUS` lacks a documented field-level mapping in the recovered export materials. It is excluded pending provenance.
