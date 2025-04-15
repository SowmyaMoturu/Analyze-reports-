# analyze_reports.py

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from embeddings_extractor import process_embeddings, compare_embeddings
from datetime import datetime
import json

def parse_reports(directory):
    all_steps = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                date_str = filename.split("_")[-1].replace(".json", "")
                date = pd.to_datetime(date_str, errors='coerce')

                for feature in data:
                    for scenario in feature.get("elements", []):
                        for step in scenario.get("steps", []):
                            all_steps.append({
                                "date": date,
                                "report": filename,
                                "feature": feature.get("name"),
                                "scenario": scenario.get("name"),
                                "step": step.get("name"),
                                "status": step.get("result", {}).get("status"),
                                "duration": step.get("result", {}).get("duration", 0),
                                "screenshot": step.get("embeddings", {}).get("screenshot") if "embeddings" in step else None
                            })
    return pd.DataFrame(all_steps)

def analyze_failures(df):
    df['is_failed'] = df['status'] != 'passed'
    fail_rate = df.groupby("scenario")['is_failed'].mean().sort_values(ascending=False)
    flaky = fail_rate[fail_rate > 0.3]
    return flaky, fail_rate

def visualize(df, output_dir):
    trend = df.groupby(["date", "status"]).size().unstack().fillna(0)
    trend.plot(kind="line", title="Pass/Fail Trend")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trend.png"))
    plt.close()

    fail_rate = df.groupby("scenario")["status"].apply(lambda x: (x != "passed").mean())
    top_fails = fail_rate.sort_values(ascending=False).head(10)
    sns.barplot(x=top_fails.values, y=top_fails.index)
    plt.title("Top Flaky Scenarios")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "flaky_scenarios.png"))
    plt.close()

def save_summary(df, flaky, fail_rate, output_dir):
    with pd.ExcelWriter(os.path.join(output_dir, "summary.xlsx")) as writer:
        df.to_excel(writer, sheet_name="All Steps", index=False)
        flaky.to_frame("fail_rate").to_excel(writer, sheet_name="Flaky Scenarios")
        fail_rate.to_frame("fail_rate").to_excel(writer, sheet_name="All Scenario Fail Rates")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--embed-screenshots", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = parse_reports(args.report_dir)
    flaky, fail_rate = analyze_failures(df)
    visualize(df, args.output_dir)
    save_summary(df, flaky, fail_rate, args.output_dir)

    if args.embed_screenshots:
        process_embeddings(df, args.output_dir)
        compare_embeddings(args.output_dir)

    print("✅ Analysis complete. Results saved to:", args.output_dir)

if __name__ == "__main__":
    main()
