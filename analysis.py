import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("visuals", exist_ok=True)

# ─── Load Data ────────────────────────────────────────────────────────────────

per_game = pd.read_csv("data/per_game_stats.csv")
salaries = pd.read_csv("data/salaries.csv")

# ─── Clean Salary Data ────────────────────────────────────────────────────────

salaries = salaries.rename(columns={
    "Unnamed: 0_level_0": "Rank",
    "Unnamed: 1_level_0": "Player",
    "Unnamed: 2_level_0": "Team",
    "Salary": "Salary_2024"
})

salaries = salaries[["Player", "Team", "Salary_2024"]].copy()
salaries = salaries.dropna(subset=["Player"])
salaries = salaries[salaries["Player"] != "Player"]

def clean_salary(val):
    try:
        return int(str(val).replace("$", "").replace(",", ""))
    except:
        return None

salaries["Salary_2024"] = salaries["Salary_2024"].apply(clean_salary)
salaries = salaries.dropna(subset=["Salary_2024"])
salaries["Player"] = salaries["Player"].str.strip()

# Remove duplicates keeping highest salary entry
salaries = salaries.sort_values("Salary_2024", ascending=False)
salaries = salaries.drop_duplicates(subset=["Player"], keep="first")

# ─── Clean Per Game Data ──────────────────────────────────────────────────────

per_game = per_game[per_game["Player"] != "Player"].copy()
per_game = per_game.dropna(subset=["Player"])
per_game["Player"] = per_game["Player"].str.strip()

numeric_cols = ["Age", "G", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
                "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%",
                "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]

for col in numeric_cols:
    if col in per_game.columns:
        per_game[col] = pd.to_numeric(per_game[col], errors="coerce")

per_game = per_game[per_game["MP"] >= 15]
per_game = per_game[per_game["G"] >= 20]

# Keep one row per player (handle traded players)
per_game = per_game.sort_values("G", ascending=False)
per_game = per_game.drop_duplicates(subset=["Player"], keep="first")

# ─── Merge ────────────────────────────────────────────────────────────────────

df = pd.merge(per_game, salaries[["Player", "Salary_2024"]], on="Player", how="inner")
df["Salary_M"] = df["Salary_2024"] / 1_000_000
print(f"Merged dataset: {len(df)} players")

# ─── Feature Engineering ──────────────────────────────────────────────────────

df["Performance_Score"] = (
    df["PTS"] * 1.0 +
    df["AST"] * 0.75 +
    df["TRB"] * 0.75 +
    df["STL"] * 1.0 +
    df["BLK"] * 1.0 -
    df["TOV"] * 0.5
)

df["Perf_Norm"] = (df["Performance_Score"] - df["Performance_Score"].min()) / \
                  (df["Performance_Score"].max() - df["Performance_Score"].min())

df["Sal_Norm"] = (df["Salary_M"] - df["Salary_M"].min()) / \
                 (df["Salary_M"].max() - df["Salary_M"].min())

df["Undervalued_Score"] = df["Perf_Norm"] - df["Sal_Norm"]
df = df.sort_values("Undervalued_Score", ascending=False).reset_index(drop=True)

print("\nTop 10 Most Undervalued Players:")
print(df[["Player", "Team", "PTS", "AST", "TRB", "Salary_M", "Undervalued_Score"]].head(10).to_string(index=False))

# ─── Plot 1: Top 15 Undervalued Players ──────────────────────────────────────

top15 = df.head(15)
fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(top15["Player"], top15["Undervalued_Score"], color="#1d428a")
ax.set_xlabel("Undervalued Score (Performance vs Salary)", fontsize=12)
ax.set_title("Top 15 Most Undervalued NBA Players - 2023/24 Season", fontsize=14, fontweight="bold")
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.savefig("visuals/top15_undervalued_players.png", dpi=150)
plt.close()
print("\nSaved: visuals/top15_undervalued_players.png")

# ─── Plot 2: Salary vs Points Per Game ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(
    df["PTS"], df["Salary_M"],
    c=df["Undervalued_Score"], cmap="coolwarm_r",
    alpha=0.7, s=60, edgecolors="none"
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Undervalued Score", fontsize=10)
ax.set_xlabel("Points Per Game", fontsize=12)
ax.set_ylabel("2024 Salary (Millions USD)", fontsize=12)
ax.set_title("NBA Player Salary vs Points Per Game - 2023/24 Season", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
plt.tight_layout()
plt.savefig("visuals/salary_vs_points.png", dpi=150)
plt.close()
print("Saved: visuals/salary_vs_points.png")

# ─── Plot 3: Performance Score by Position ────────────────────────────────────

pos_df = df[df["Pos"].notna()].copy()
pos_df["Pos"] = pos_df["Pos"].str.split("-").str[0]
pos_order = ["PG", "SG", "SF", "PF", "C"]
pos_df = pos_df[pos_df["Pos"].isin(pos_order)]

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=pos_df, x="Pos", y="Performance_Score", order=pos_order, palette="Blues", ax=ax)
ax.set_xlabel("Position", fontsize=12)
ax.set_ylabel("Performance Score", fontsize=12)
ax.set_title("Performance Score Distribution by Position - 2023/24 Season", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("visuals/performance_by_position.png", dpi=150)
plt.close()
print("Saved: visuals/performance_by_position.png")

# ─── Plot 4: Top 10 Overvalued Players ───────────────────────────────────────

bottom10 = df.nsmallest(10, "Undervalued_Score")
fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(bottom10["Player"], bottom10["Undervalued_Score"], color="#c8102e")
ax.set_xlabel("Undervalued Score (lower = more overvalued)", fontsize=12)
ax.set_title("Top 10 Most Overvalued NBA Players - 2023/24 Season", fontsize=14, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("visuals/top10_overvalued_players.png", dpi=150)
plt.close()
print("Saved: visuals/top10_overvalued_players.png")

# ─── Machine Learning: Predict Player Salary ─────────────────────────────────

print("\nTraining Random Forest model to predict player salary...")

features = ["Age", "G", "MP", "PTS", "AST", "TRB", "STL", "BLK", "TOV",
            "FG%", "3P%", "FT%", "eFG%", "Performance_Score"]

ml_df = df[features + ["Salary_M", "Pos"]].copy()

le = LabelEncoder()
ml_df["Pos"] = ml_df["Pos"].str.split("-").str[0]
ml_df["Pos_Encoded"] = le.fit_transform(ml_df["Pos"].fillna("SF"))
ml_df = ml_df.dropna()

X = ml_df[features + ["Pos_Encoded"]]
y = ml_df["Salary_M"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"  Mean Absolute Error : ${mae:.2f}M")
print(f"  R-squared Score     : {r2:.3f}")

# ─── Plot 5: Feature Importance ───────────────────────────────────────────────

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(importances.index, importances.values, color="#1d428a")
ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_title("Random Forest - Feature Importance for Salary Prediction", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("visuals/feature_importance.png", dpi=150)
plt.close()
print("Saved: visuals/feature_importance.png")

# ─── Plot 6: Actual vs Predicted Salary ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(y_test, y_pred, alpha=0.6, color="#1d428a", edgecolors="none")
ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", linewidth=1.5, label="Perfect Prediction")
ax.set_xlabel("Actual Salary (Millions USD)", fontsize=12)
ax.set_ylabel("Predicted Salary (Millions USD)", fontsize=12)
ax.set_title("Actual vs Predicted NBA Player Salary", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
ax.legend()
plt.tight_layout()
plt.savefig("visuals/actual_vs_predicted_salary.png", dpi=150)
plt.close()
print("Saved: visuals/actual_vs_predicted_salary.png")

# ─── Summary ──────────────────────────────────────────────────────────────────

print("\n--- Summary Statistics ---")
print(f"Total players analyzed : {len(df)}")
print(f"Average salary         : ${df['Salary_M'].mean():.2f}M")
print(f"Average points per game: {df['PTS'].mean():.1f}")
print(f"Average performance    : {df['Performance_Score'].mean():.2f}")
print("\nAnalysis complete. All visuals saved to visuals/ folder.")