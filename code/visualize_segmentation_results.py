import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from math import pi

# Suppress SettingWithCopy warnings for cleaner logs
pd.options.mode.chained_assignment = None

# Configuration
SEG_DIR = Path(__file__).parent / "output" / "segmentation"
FIG_DIR = SEG_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Load data
corr = pd.read_csv(SEG_DIR /                "correlation_matrix.csv", index_col=0)
feat_imp = pd.read_csv(SEG_DIR /            "feature_importances.csv")
cluster_gdf = gpd.read_file(SEG_DIR /       "clustered_tracts_10.geojson")
pca = pd.read_csv(SEG_DIR /                 "pca_components.csv", index_col=0)
clust = pd.read_csv(SEG_DIR /               "tract_clusters_10.csv", index_col="GEOID")
doppel_nn = pd.read_csv(SEG_DIR /           "doppelganger_nn.csv")
# Other outputs
cluster_profiles = pd.read_csv(SEG_DIR /    "cluster_profiles_10.csv")
conf_mat = pd.read_csv(SEG_DIR /            "confusion_matrix_10_clusters.csv", header=None)
missing_report = pd.read_csv(SEG_DIR /      "missing_data_analysis.csv")
outliers = pd.read_csv(SEG_DIR /            "isolation_forest_outliers.csv", dtype={"GEOID": str})


def plot_correlation_heatmap():
    # Read and sort feature importance, get top 10 features
    feat_imp_df = pd.read_csv(SEG_DIR / "feature_importances.csv")
    top_features = feat_imp_df.sort_values("Importance", ascending=False)["Feature"].head(10).tolist()

    # Map features to layman names
    feature_name_mapping = {
        "health_copd_places": "COPD prevalence",
        "health_indeplive_places": "Independent living difficulty",
        "health_selfcare_places": "Self-care difficulty",
        "health_ghlth_places": "Poor general health",
        "health_arthritis_places": "Arthritis prevalence",
        "health_disability_places": "Disability rate",
        "health_mobility_places": "Mobility difficulty",
        "health_cancer_places": "Cancer diagnosis rate",
        "health_dental_places": "No dental visit rate",
        "health_chd_places": "Coronary heart disease rate",
    }

    # Apply mapping if feature exists
    display_labels = [feature_name_mapping.get(f, f) for f in top_features]

    # Subset the correlation matrix
    corr_subset = corr.loc[top_features, top_features]
    mask = np.triu(np.ones_like(corr_subset, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_subset, cmap="coolwarm", center=0, annot=False,
                xticklabels=display_labels, yticklabels=display_labels, mask=mask)
    plt.xticks(rotation=45, ha='right')    
    plt.title("Feature Correlation Matrix (Top 10 Features)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()


def plot_feature_importance():
    # Map technical names to layman-friendly labels
    feature_name_mapping = {
        "health_copd_places": "COPD prevalence",
        "health_indeplive_places": "Independent living difficulty",
        "health_selfcare_places": "Self-care difficulty",
        "health_ghlth_places": "Poor general health",
        "health_arthritis_places": "Arthritis prevalence",
        "health_disability_places": "Disability rate",
        "health_mobility_places": "Mobility difficulty",
        "health_cancer_places": "Cancer diagnosis rate",
        "health_dental_places": "No dental visit rate",
        "health_chd_places": "Coronary heart disease rate",
    }

    top = feat_imp.sort_values("Importance", ascending=False).head(10).copy()
    top["Feature"] = top["Feature"].map(lambda f: feature_name_mapping.get(f, f))  # Apply mapping

    plt.figure(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=top)
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance_bar.png", dpi=300)
    plt.close()


def plot_cluster_map():
    exclude_statefps = {'02', '15', '60', '66', '69', '72', '78'}
    if 'STATEFP' in cluster_gdf.columns:
        gdf_plot = cluster_gdf[~cluster_gdf['STATEFP'].isin(exclude_statefps)]
    else:
        gdf_plot = cluster_gdf.copy()
        print("Warning: 'STATEFP' column not found. No state filtering applied.")

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    gdf_plot.plot(column="cluster", categorical=True, legend=True, ax=ax, cmap="tab20", markersize=1)
    ax.axis("off")
    plt.title("Census Tract Clusters (Contiguous US)")
    ax.set_xlim([-125, -66])
    ax.set_ylim([24, 50])

    # Move legend to bottom right
    leg = ax.get_legend()
    leg.set_bbox_to_anchor((1, 0.2))  # (x, y) — 0.2 = 20% from bottom

    plt.tight_layout()
    plt.savefig(FIG_DIR / "cluster_map.png", dpi=4500)
    plt.close()


def plot_cluster_sizes():
    """Bar chart of number of tracts in each cluster."""
    counts_series = clust["cluster"].value_counts().sort_index()
    counts_df = counts_series.rename_axis("cluster").reset_index(name="count")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=counts_df, y="count", x="cluster", palette=sns.color_palette("Reds_r"))
    plt.title("Number of Tracts per Cluster")
    plt.ylabel("Count")
    plt.xlabel("Cluster")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cluster_size_bar.png", dpi=300)
    plt.close()


def plot_confusion_heatmap():
    if conf_mat.empty:
        return

    # Set row and column labels (1–10 instead of 0–9)
    labels = list(range(1, 11))  # 1 to 10 inclusive

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title("XGBoost Confusion Matrix (10 Clusters)")
    plt.xlabel("Predicted Cluster")
    plt.ylabel("Actual Cluster")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix_heatmap.png", dpi=300)
    plt.close()


def plot_missing_data_heatmap():
    if missing_report.empty:
        return

    df = missing_report.copy()

    if {"Column", "Missing_Percent"}.issubset(df.columns):
        bar_df = df[["Column", "Missing_Percent"]].rename(columns={"Column": "feature", "Missing_Percent": "missing_pct"})
    else:
        bar_df = df.iloc[:, :2]
        bar_df.columns = ["feature", "missing_pct"]

    bar_df = bar_df.sort_values("missing_pct", ascending=False).head(10)
    feature_name_mapping = {
        "health_copd_places": "COPD prevalence",
        "health_indeplive_places": "Independent living difficulty",
        "health_selfcare_places": "Self-care difficulty",
        "health_ghlth_places": "Poor general health",
        "health_arthritis_places": "Arthritis prevalence",
        "health_disability_places": "Disability rate",
        "health_mobility_places": "Mobility difficulty",
        "health_cancer_places": "Cancer diagnosis rate",
        "health_dental_places": "No dental visit rate",
        "health_chd_places": "Coronary heart disease rate",
        "health_housinsecu_places": "Housing insecurity rate",
        "health_foodinsecu_places": "Food insecurity rate",
        "health_foodstamp_places": "Food stamp rate",
        "health_emotionspt_places": "Emotional support rate",
        "health_lacktrpt_places": "Lack of transportation rate",
        "health_isolation_places": "Isolation rate",
        "health_shututility_places": "Shut-in rate",
        "health_cholscreen_places": "Cholesterol screening rate",
        "health_bphigh_places": "High blood pressure rate",
        "health_bpmed_places": "High blood pressure medication rate"
    }
    bar_df["feature"] = bar_df["feature"].map(lambda f: feature_name_mapping.get(f, f))

    bar_df["missing_pct"] = bar_df["missing_pct"] / 100

    plt.figure(figsize=(8, 10))
    ax = sns.barplot(data=bar_df, y="feature", x="missing_pct", palette=sns.color_palette("Reds_r"))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.1%}',
                    (p.get_width(), p.get_y() + p.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=9)

    plt.title("Missing Data Fraction by Feature (pre-imputation)")
    plt.xlabel("")
    plt.xticks([])  # Hide x-axis ticks
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "missing_data_bar.png", dpi=300)
    plt.close()


def plot_radar_cluster_profiles():
    if cluster_profiles.empty:
        return

    df = cluster_profiles.copy()
    df.index = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    df.index.name = "feature"

    top_vars = df.var(axis=1).sort_values(ascending=False).head(5).index.tolist()
    df = df.loc[top_vars]
    df = df.transpose()
    df_norm = (df - df.min()) / (df.max() - df.min())

    categories = df_norm.columns.tolist()
    feature_name_mapping = {
        "health_copd_places": "COPD prevalence",
        "health_indeplive_places": "Independent living difficulty",
        "health_selfcare_places": "Self-care difficulty",
        "health_ghlth_places": "Poor general health",
        "health_arthritis_places": "Arthritis prevalence",
        "health_disability_places": "Disability rate",
        "health_mobility_places": "Mobility difficulty",
        "health_cancer_places": "Cancer diagnosis rate",
        "health_dental_places": "No dental visit rate",
        "health_chd_places": "Coronary heart disease rate",
        "higher_education_pct": "Higher education rate",
        "masters_degree": "Master's degree rate",
        "bachelors_degree": "Bachelor's degree rate"
    }
    categories = [feature_name_mapping.get(c, c) for c in categories]

    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(df_norm)))

    for idx, (i, row) in enumerate(df_norm.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=f"Cluster {i}", color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)

    ax.set_yticklabels([])  # Hide radial grid labels
    ax.grid(color="gray", linestyle="dotted", linewidth=0.7)

    plt.title("Normalized Cluster Profiles (Top 5 Features)", size=16, y=1.15)
    plt.legend(bbox_to_anchor=(1.2, 0.9), frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cluster_profile_radar.png", dpi=300)
    plt.close()


def plot_outlier_map():
    if outliers.empty:
        return

    exclude_statefps = {'02', '15', '60', '66', '69', '72', '78'}  # AK, HI, territories
    if 'STATEFP' in cluster_gdf.columns:
        gdf_plot = cluster_gdf[~cluster_gdf['STATEFP'].isin(exclude_statefps)].copy()
    else:
        gdf_plot = cluster_gdf.copy()
        print("Warning: 'STATEFP' column not found. No state filtering applied.")

    # Merge with filtered GeoDataFrame
    outlier_geo = gdf_plot.merge(outliers[["GEOID", "Outlier_Score"]], on="GEOID", how="left") # 'how=inner' if current looks poor/doesn't make sense

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    gdf_plot.plot(ax=ax, color="#eeeeee", linewidth=0.2, edgecolor="black")
    outlier_geo.plot(ax=ax, column="Outlier_Score", cmap="inferno_r", legend=True, markersize=1, legend_kwds={"shrink": 0.25})
    ax.axis("off")
    plt.title("Isolation Forest Outlier Scores by Tract (Contiguous US)")
    ax.set_xlim([-125, -66])
    ax.set_ylim([24, 50])

    plt.tight_layout()
    plt.savefig(FIG_DIR / "outlier_map.png", dpi=4500)
    plt.close()


def plot_doppelganger_similarity():
    plt.figure(figsize=(8, 4))
    ax = sns.histplot(doppel_nn["similarity"], bins=50, kde=True)
    plt.title("Doppelganger Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    ax.set_yscale('symlog', linthresh=1)

    # Add count labels
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.text(patch.get_x() - 5, height, f"{int(height)}",
                    ha="right", va="bottom", fontsize=6, rotation=45)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "doppel_similarity_hist_log_scale.png", dpi=300)
    plt.close()


def main():
    plot_correlation_heatmap()
    plot_feature_importance()
    plot_cluster_map()
    plot_cluster_sizes()
    plot_confusion_heatmap()
    plot_missing_data_heatmap()
    plot_radar_cluster_profiles()
    plot_outlier_map()
    plot_doppelganger_similarity()
    print(f"Figures saved to {FIG_DIR}")

if __name__ == "__main__":
    main()