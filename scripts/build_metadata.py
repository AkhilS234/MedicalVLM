import os
import pandas as pd

rows = []

# --- Montgomery ---
image_dir = "data/raw/MontgomerySet/CXR_png"
text_dir = "data/raw/MontgomerySet/ClinicalReadings"

for filename in os.listdir(image_dir):
    if not filename.endswith(".png"):
        continue

    image_path = os.path.join(image_dir, filename)
    stem = filename.replace(".png", "")
    txt_path = os.path.join(text_dir, stem + ".txt")
    label = "normal" if stem.endswith("_0") else "tuberculosis"

    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        sex = next((l for l in lines if "Sex" in l), "").replace("Patient's Sex:", "").strip()
        age = next((l for l in lines if "Age" in l), "").replace("Patient's Age:", "").strip()
        diagnosis = lines[-1].strip()
        if diagnosis.lower() == "normal":
            text = f"Normal chest X-ray. No active disease. Patient: {sex}, age {age}."
        else:
            text = f"Abnormal chest X-ray. Finding: {diagnosis}. Patient: {sex}, age {age}."
    else:
        text = f"Chest X-ray. Diagnosis: {label}."

    rows.append({"image_path": image_path, "text": text})

# --- Shenzhen ---
shenzhen_csv = "data/raw/ChinaSet/shenzhen_metadata.csv"
shenzhen_image_dir = "data/raw/ChinaSet/images/images"
df_shenzhen = pd.read_csv(shenzhen_csv)

for _, row in df_shenzhen.iterrows():
    image_path = os.path.join(shenzhen_image_dir, row["study_id"])
    if not os.path.exists(image_path):
        continue

    sex = str(row["sex"])
    age = str(row["age"])
    diagnosis = str(row["findings"]).strip()

    if diagnosis.lower() == "normal":
        text = f"Normal chest X-ray. No active disease. Patient: {sex}, age {age}."
    else:
        text = f"Abnormal chest X-ray. Finding: {diagnosis}. Patient: {sex}, age {age}."

    rows.append({"image_path": image_path, "text": text})

if diagnosis.lower() == "normal":
    text = f"Normal chest X-ray with clear lung fields. No active disease detected. {sex} patient, {age} years old. No infiltrates, effusions, or masses identified."
else:
    text = f"Abnormal chest X-ray. Radiological finding: {diagnosis}. {sex} patient, {age} years old."

df = pd.DataFrame(rows)
df.to_csv("data/processed/metadata.csv", index=False)
print(f"Saved {len(df)} rows")
print(f"Montgomery + Shenzhen combined")
print(df.head())