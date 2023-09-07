import os
import pandas as pd


def logging(PROJECT_NAME, lapse_time, results):
    results = [PROJECT_NAME, lapse_time] + results

    result_df = pd.DataFrame(
        [results],
        columns=[
            "Project",
            "Lapse_time",
            "DAVIS_AUROC",
            "DAIVS_AUPRC",
            "DAVIS_SEN",
            "DAVIS_SPEC",
            "BINDING_AUROC",
            "BINDING_AUPRC",
            "BINDING_SEN",
            "BINDING_SPEC",
            "BIOSNAP_AUROC",
            "BIOSNAP_AUPRC",
            "BIOSNAP_SEN",
            "BIOSNAP_SPEC",
        ],
    )

    os.makedirs("results", exist_ok=True)
    if os.path.exists("./results/results.csv"):
        result_df_orig = pd.read_csv("./results/results.csv")
        result_df = pd.concat([result_df_orig, result_df], axis=0)
        result_df.to_csv("./results/results.csv", index=False)
    else:
        result_df.to_csv("./results/results.csv", index=False)
