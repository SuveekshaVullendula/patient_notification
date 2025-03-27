import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import datetime
import smtplib
from email.mime.text import MIMEText

# Simulated Notification Function (Replace with actual SMS/Email service)
def send_notification(message, recipient_email):
    print(f"[NOTIFICATION]: {message}")
    # Uncomment below for real email sending (use your credentials)
    msg = MIMEText(message)
    msg['Subject'] = 'Alzheimers Patient Alert'
    msg['From'] = 'suveeksha287@gmail.com'
    msg['To'] = recipient_email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('suveeksha287@gmail.com', 'mazl hwol acbw bith')
    server.sendmail('suveeksha287@gmail.com', recipient_email, msg.as_string())
    server.quit()

# Load Dataset (assumes 'dataset.csv' already exists)
df = pd.read_csv("dataset.csv")

# Feature Engineering
df['LastSeenHour'] = np.random.randint(0, 24, size=len(df))

def categorize_time_of_day(hour):
    if 6 <= hour < 12:
        return 0  # Morning
    elif 12 <= hour < 18:
        return 1  # Afternoon
    elif 18 <= hour < 24:
        return 2  # Evening
    else:
        return 3  # Night

df['TimeOfDay'] = df['LastSeenHour'].apply(lambda x: categorize_time_of_day(x) if pd.notnull(x) else np.random.randint(0, 4))

df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
df['Education'] = LabelEncoder().fit_transform(df['EducationLevel'])

# Feature Selection and Risk Level Calculation
features = ['Age', 'Gender', 'EducationLevel', 'SleepQuality', 'Depression', 'PhysicalActivity',
            'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion', 'CholesterolHDL']

def determine_risk(radius):
    if radius < 500:
        return 0  # Low risk
    elif 500 <= radius < 1000:
        return 1  # Medium risk
    else:
        return 2  # High risk

df['RiskLevel'] = df['Radius'].apply(determine_risk)

X = df[features]
y = df['RiskLevel']

X.fillna(X.median(), inplace=True)

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
print("Classification Report:\n", classification_report(y_test, model.predict(X_test)))

# For tracking daily notifications
notified_patients = set()

# Process and Notify for Multiple Patients
def process_patients(patients_data):
    today = datetime.date.today()

    for patient in patients_data:
        patient_id = patient['PatientID']
        recipient_email = patient['FamilyEmail']

        # Skip sending if patient already notified today
        if (patient_id, today) in notified_patients:
            continue

        # Prepare new patient DataFrame and scale features
        patient_features = {k: patient[k] for k in features}
        new_patient_df = pd.DataFrame([patient_features])
        new_patient_df = pd.DataFrame(scaler.transform(new_patient_df), columns=new_patient_df.columns)

        # Make prediction and notify based on risk level
        prediction = model.predict(new_patient_df)[0]
        last_seen_hour = np.random.randint(0, 24)  # Placeholder logic for testing

        if prediction == 2:  # High Risk
            notify_time = max(0, last_seen_hour - 1)
            message = f"High Alert: Patient {patient_id} may cross at {last_seen_hour}:00."
            send_notification(message, recipient_email)
        elif prediction == 1:  # Medium Risk
            notify_time = max(0, last_seen_hour - 2)
            message = f"Warning: Patient {patient_id} might cross at {last_seen_hour}:00."
            send_notification(message, recipient_email)

        notified_patients.add((patient_id, today))

# Test Patients Data
test_patients = [
    {'PatientID': 104, 'Age': 78, 'Gender': 1, 'EducationLevel': 3, 'SleepQuality': 2, 'Depression': 0, 'PhysicalActivity': 1,
     'MemoryComplaints': 1, 'BehavioralProblems': 0, 'ADL': 3, 'Confusion': 1, 'CholesterolHDL': 45, 'FamilyEmail': 'suveekshavullendula@gmail.com'},
    {'PatientID': 105, 'Age': 85, 'Gender': 0, 'EducationLevel': 2, 'SleepQuality': 3, 'Depression': 1, 'PhysicalActivity': 0,
     'MemoryComplaints': 1, 'BehavioralProblems': 1, 'ADL': 2, 'Confusion': 1, 'CholesterolHDL': 50, 'FamilyEmail': 'mokshithareddy224@gmail.com'},
    {'PatientID': 106, 'Age': 72, 'Gender': 1, 'EducationLevel': 4, 'SleepQuality': 1, 'Depression': 0, 'PhysicalActivity': 1,
     'MemoryComplaints': 0, 'BehavioralProblems': 2, 'ADL': 4, 'Confusion': 0, 'CholesterolHDL': 39, 'FamilyEmail': 'deekshithanadikattu@gmail.com'},
    {'PatientID': 107, 'Age': 80, 'Gender': 0, 'EducationLevel': 1, 'SleepQuality': 4, 'Depression': 1, 'PhysicalActivity': 0,
     'MemoryComplaints': 1, 'BehavioralProblems': 1, 'ADL': 1, 'Confusion': 1, 'CholesterolHDL': 42, 'FamilyEmail': 'rasagnauyyala@gmail.com'},
    {'PatientID': 108, 'Age': 67, 'Gender': 1, 'EducationLevel': 3, 'SleepQuality': 3, 'Depression': 0, 'PhysicalActivity': 1,
     'MemoryComplaints': 0, 'BehavioralProblems': 0, 'ADL': 5, 'Confusion': 1, 'CholesterolHDL': 48, 'FamilyEmail': 'yelalanikitha@gmail.com'}
]

# Process multiple patients
process_patients(test_patients)
