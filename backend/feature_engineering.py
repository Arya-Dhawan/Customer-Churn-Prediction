import pandas as pd

def engineer_features(df):

    # Tenure groups
    def tenure_group(t):
        if t <= 12:
            return "0-1yr"
        elif t <= 24:
            return "1-2yr"
        elif t <= 48:
            return "2-4yr"
        else:
            return "4-6yr"

    df["tenure_group"] = df["tenure"].apply(tenure_group)

    # Number of services
    services = [
        "PhoneService","MultipleLines","OnlineSecurity",
        "OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies"
    ]

    df["num_services"] = (df[services] == "Yes").sum(axis=1)

    return df