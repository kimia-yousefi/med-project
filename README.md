Surrogate Mother Matching System
📌 Project Description

This project is a decision support system that helps intended parents find the most suitable surrogate mother.
The algorithm uses:

Static features of donors (medical, psychological, physical, supportive factors).

Dynamic features from families (location, cultural/religious preferences, education, income).

The system ranks donors using:

Weighted scoring for static features (closer to the ideal gets lower error → better).

Euclidean distance for dynamic features (closer preferences → better).

Final ranking is based on the lowest combined distance score.

The app is implemented in Python (Flask + Pandas) and reads donor data from an Excel file.

🧩 Algorithm Overview

Static Features (donors only):

Age (mapped to an ideal scoring table).

BMI.

Uterus health (anatomy, endometrium, hormonal factors).

Pregnancy history (successful births, miscarriages, infertility, complications, etc.).

Physical health (chronic illness, infections, cardiovascular, kidney, immunology).

Psychological health (MMPI test).

Family support (MSPSS test).

Each category has a predefined weight (e.g., Age 22%, BMI 13%, etc.).

👉 Normalized & weighted → produces a static score.

Dynamic Features (family + donor):

Location (same city, same region, far).

Education level.

Marriage status.

Socio-cultural compatibility: Ethnicity, Religion, Faith (importance levels defined by families).

Income similarity.

👉 Euclidean distance is computed between donor and family preferences.

Final Matching:

Total score = Static Score + Dynamic Distance.

Donors are ranked in ascending order (lowest = best match).

📂 Data Structure
Donor Excel Example (donors.xlsx)

Each donor has static + dynamic features:

donor_id	age	BMI	anatomy	endometrium	hormonal_factor	para	abort	infertility	...	religion	faith	ethnicity	location	education	marriage	income_level	...

👉 Families do not appear in the Excel file. Their info is entered through the web form (Flask).


🔮 Future Work

Add more family-side preferences (health history, psychological factors).

Improve UI/UX.

Extend matching algorithm with ML models (e.g., KNN, Random Forest) for comparison.



Developed by Kimia Yousefifard
