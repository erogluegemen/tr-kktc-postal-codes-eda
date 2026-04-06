"""
CMP5101 Data Mining - Assignment I
Exploratory Data Analysis on TR/KKTC Postal Codes Dataset
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

os.makedirs("figures", exist_ok=True)

sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams.update({"figure.dpi": 150, "figure.figsize": (10, 5)})

print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

df = pd.read_csv(
    "tr_kktc_postal_codes.csv",
    sep=";",
    decimal=",",
    encoding="utf-8",
    dtype={"PK": str},   # keep postal code as string to preserve leading zeros
)

df.columns = ["il", "ilce", "semt", "mahalle", "PK", "lat", "lon"]

print(f"Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 70)
print("2. FEATURE TYPES")
print("=" * 70)

type_map = {
    "il":      "Categorical (nominal) – Province name",
    "ilce":    "Categorical (nominal) – District name",
    "semt":    "Categorical (nominal) – Township / town",
    "mahalle": "Categorical (nominal) – Neighbourhood / village",
    "PK":      "Quasi-categorical (ordinal) – Postal code",
    "lat":     "Numeric (continuous) – Latitude",
    "lon":     "Numeric (continuous) – Longitude",
}

print(f"\n{'Column':<10} {'Pandas dtype':<15} {'Semantic type'}")
print("-" * 70)
for col in df.columns:
    print(f"{col:<10} {str(df[col].dtype):<15} {type_map[col]}")

print("\n" + "=" * 70)
print("3. SUMMARY STATISTICS")
print("=" * 70)

cat_cols = ["il", "ilce", "semt", "mahalle"]
print("\nCategorical columns – unique value counts:")
for col in cat_cols:
    print(f"  {col:<10}: {df[col].nunique():>6,} unique values")

print("\nTop 10 provinces by record count:")
print(df["il"].value_counts().head(10).to_string())

pk_numeric = pd.to_numeric(df["PK"], errors="coerce")
print(f"\nPostal code (PK):")
print(f"  Total unique  : {df['PK'].nunique():,}")
print(f"  Min           : {pk_numeric.min()}")
print(f"  Max           : {pk_numeric.max()}")
print(f"  Range         : {pk_numeric.min()} – {pk_numeric.max()}")

tr_df = df[df["il"] != "KKTC"].copy()
kktc_df = df[df["il"] == "KKTC"].copy()

print(f"\nNumeric columns (TR rows only, n={len(tr_df):,}):")
print(tr_df[["lat", "lon"]].describe().round(4))

print(f"\nNumeric columns (KKTC rows, n={len(kktc_df):,}):")
print(kktc_df[["lat", "lon"]].describe().round(4))

print("\n" + "=" * 70)
print("4. MISSING DATA")
print("=" * 70)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing count": missing, "Missing %": missing_pct})
print(missing_df)

print("\nEmpty-string counts (categorical columns):")
for col in cat_cols:
    n_empty = (df[col].astype(str).str.strip() == "").sum()
    print(f"  {col:<10}: {n_empty}")

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#e74c3c" if p > 0 else "#2ecc71" for p in missing_pct]
bars = ax.bar(missing_df.index, missing_df["Missing %"], color=colors, edgecolor="white")
ax.set_title("Missing Data Percentage per Column", fontsize=13)
ax.set_ylabel("Missing (%)")
ax.set_xlabel("Column")
for bar, val in zip(bars, missing_df["Missing %"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig01_missing_data.png")
plt.close()
print("\n[Saved] figures/fig01_missing_data.png")

print("\n" + "=" * 70)
print("5. DUPLICATE DETECTION")
print("=" * 70)

full_dups = df.duplicated().sum()
loc_dups = df.duplicated(subset=["il", "ilce", "semt", "mahalle"]).sum()
pk_mah_dups = df.duplicated(subset=["PK", "mahalle"]).sum()

print(f"  Full-row duplicates                     : {full_dups:,}")
print(f"  Location duplicates (il+ilce+semt+mah)  : {loc_dups:,}")
print(f"  PK+Neighbourhood duplicates             : {pk_mah_dups:,}")

if loc_dups > 0:
    print("\n  Sample location duplicates:")
    dup_mask = df.duplicated(subset=["il", "ilce", "semt", "mahalle"], keep=False)
    print(df[dup_mask].sort_values(["il", "ilce", "semt", "mahalle"]).head(10).to_string(index=False))

print("\n" + "=" * 70)
print("6. HISTOGRAMS & BOX PLOTS")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(pk_numeric.dropna(), bins=80, color="#3498db", edgecolor="white", alpha=0.85)
ax.set_title("Distribution of Postal Codes (PK)", fontsize=13)
ax.set_xlabel("Postal Code")
ax.set_ylabel("Frequency")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("figures/fig02_pk_histogram.png")
plt.close()
print("[Saved] figures/fig02_pk_histogram.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
tr_lat = tr_df["lat"].dropna()
axes[0].hist(tr_lat, bins=50, color="#2ecc71", edgecolor="white", alpha=0.85)
axes[0].set_title("Latitude Distribution (TR)")
axes[0].set_xlabel("Latitude")
axes[0].set_ylabel("Frequency")
axes[1].boxplot(tr_lat, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#2ecc71", color="#27ae60"))
axes[1].set_title("Latitude Boxplot (TR)")
axes[1].set_ylabel("Latitude")
plt.suptitle("Latitude – Turkey Records", fontsize=13)
plt.tight_layout()
plt.savefig("figures/fig03_latitude_dist.png")
plt.close()
print("[Saved] figures/fig03_latitude_dist.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
tr_lon = tr_df["lon"].dropna()
axes[0].hist(tr_lon, bins=50, color="#e67e22", edgecolor="white", alpha=0.85)
axes[0].set_title("Longitude Distribution (TR)")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Frequency")
axes[1].boxplot(tr_lon, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#e67e22", color="#d35400"))
axes[1].set_title("Longitude Boxplot (TR)")
axes[1].set_ylabel("Longitude")
plt.suptitle("Longitude – Turkey Records", fontsize=13)
plt.tight_layout()
plt.savefig("figures/fig04_longitude_dist.png")
plt.close()
print("[Saved] figures/fig04_longitude_dist.png")

top_il = df["il"].value_counts().head(20)
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(top_il.index, top_il.values, color=sns.color_palette("tab20", 20), edgecolor="white")
ax.set_title("Top 20 Provinces by Number of Neighbourhood Records", fontsize=13)
ax.set_xlabel("Province (İl)")
ax.set_ylabel("Record Count")
plt.xticks(rotation=45, ha="right")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig("figures/fig05_top_provinces.png")
plt.close()
print("[Saved] figures/fig05_top_provinces.png")

top_ilce = df["ilce"].value_counts().head(20)
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(top_ilce.index, top_ilce.values, color=sns.color_palette("tab20b", 20), edgecolor="white")
ax.set_title("Top 20 Districts by Number of Neighbourhood Records", fontsize=13)
ax.set_xlabel("District (İlçe)")
ax.set_ylabel("Record Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("figures/fig06_top_districts.png")
plt.close()
print("[Saved] figures/fig06_top_districts.png")

mah_per_pk = df.groupby("PK")["mahalle"].count()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(mah_per_pk, bins=60, color="#9b59b6", edgecolor="white", alpha=0.85)
axes[0].set_title("Neighbourhood Count per Postal Code")
axes[0].set_xlabel("# Neighbourhoods sharing one PK")
axes[0].set_ylabel("Frequency")
axes[1].boxplot(mah_per_pk, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#9b59b6", color="#8e44ad"))
axes[1].set_title("Neighbourhoods per PK – Boxplot")
axes[1].set_ylabel("Count")
plt.suptitle("Postal Code Sharing: How Many Neighbourhoods Share One PK?", fontsize=12)
plt.tight_layout()
plt.savefig("figures/fig07_mah_per_pk.png")
plt.close()
print("[Saved] figures/fig07_mah_per_pk.png")

print("\n" + "=" * 70)
print("7. RELATIONSHIPS BETWEEN FEATURES")
print("=" * 70)

sample = tr_df[["lat", "lon", "il"]].dropna().sample(min(15000, len(tr_df)), random_state=42)
fig, ax = plt.subplots(figsize=(11, 7))
scatter = ax.scatter(sample["lon"], sample["lat"], s=1, alpha=0.4, color="#2980b9")
ax.set_title("Geographic Distribution of Records (Latitude vs Longitude) – TR", fontsize=13)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_xlim(25, 45)
ax.set_ylim(35, 43)
plt.tight_layout()
plt.savefig("figures/fig08_geo_scatter.png", dpi=150)
plt.close()
print("[Saved] figures/fig08_geo_scatter.png")

region_counts = pd.Series({"Turkey (TR)": len(tr_df), "KKTC": len(kktc_df)})
fig, ax = plt.subplots(figsize=(6, 4))
colors_reg = ["#3498db", "#e74c3c"]
ax.bar(region_counts.index, region_counts.values, color=colors_reg, edgecolor="white", width=0.5)
ax.set_title("Record Count: Turkey vs KKTC", fontsize=13)
ax.set_ylabel("Record Count")
for i, (label, val) in enumerate(region_counts.items()):
    ax.text(i, val + 50, f"{val:,}", ha="center", va="bottom", fontsize=11)
plt.tight_layout()
plt.savefig("figures/fig09_tr_vs_kktc.png")
plt.close()
print("[Saved] figures/fig09_tr_vs_kktc.png")

pk_numeric_series = pk_numeric.rename("PK_num")
num_df = pd.concat([tr_df[["lat", "lon"]].reset_index(drop=True),
                    pk_numeric_series[tr_df.index].reset_index(drop=True)], axis=1).dropna()
corr = num_df.corr()
print("\nCorrelation matrix (lat, lon, PK_num) – TR rows:")
print(corr.round(4))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix: Latitude, Longitude, Postal Code", fontsize=12)
plt.tight_layout()
plt.savefig("figures/fig10_correlation_heatmap.png")
plt.close()
print("[Saved] figures/fig10_correlation_heatmap.png")

top15_il = df["il"].value_counts().head(15).index.tolist()
sub = tr_df[tr_df["il"].isin(top15_il)].dropna(subset=["lat"])
fig, ax = plt.subplots(figsize=(14, 5))
order = sub.groupby("il")["lat"].median().sort_values().index
sns.boxplot(data=sub, x="il", y="lat", order=order, palette="tab20", ax=ax,
            flierprops=dict(marker=".", markersize=2, alpha=0.3))
ax.set_title("Latitude Distribution by Province (Top 15 – sorted by median latitude)", fontsize=12)
ax.set_xlabel("Province")
ax.set_ylabel("Latitude")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("figures/fig11_lat_by_province.png")
plt.close()
print("[Saved] figures/fig11_lat_by_province.png")

print("\n" + "=" * 70)
print("8. ANOMALY / ERROR DETECTION")
print("=" * 70)

TR_PK_MIN, TR_PK_MAX = 1000, 81999
KKTC_PK_MIN, KKTC_PK_MAX = 99000, 99999

tr_pk_num = pk_numeric[tr_df.index]
kktc_pk_num = pk_numeric[kktc_df.index]

tr_out_of_range = tr_pk_num[(tr_pk_num < TR_PK_MIN) | (tr_pk_num > TR_PK_MAX)]
kktc_out_of_range = kktc_pk_num[(kktc_pk_num < KKTC_PK_MIN) | (kktc_pk_num > KKTC_PK_MAX)]

print(f"\nTR postal codes outside [{TR_PK_MIN}–{TR_PK_MAX}]: {len(tr_out_of_range):,}")
if len(tr_out_of_range) > 0:
    print(df.loc[tr_out_of_range.index, ["il","ilce","PK"]].head(10).to_string(index=False))

print(f"KKTC postal codes outside [{KKTC_PK_MIN}–{KKTC_PK_MAX}]: {len(kktc_out_of_range):,}")
if len(kktc_out_of_range) > 0:
    print(df.loc[kktc_out_of_range.index, ["il","ilce","PK"]].head(10).to_string(index=False))

LAT_MIN, LAT_MAX = 35.0, 43.0
LON_MIN, LON_MAX = 25.0, 45.0

coord_df = tr_df.dropna(subset=["lat", "lon"])
bad_lat = coord_df[(coord_df["lat"] < LAT_MIN) | (coord_df["lat"] > LAT_MAX)]
bad_lon = coord_df[(coord_df["lon"] < LON_MIN) | (coord_df["lon"] > LON_MAX)]

print(f"\nRecords with latitude outside [{LAT_MIN}–{LAT_MAX}]: {len(bad_lat):,}")
if len(bad_lat) > 0:
    print(bad_lat[["il", "ilce", "mahalle", "lat", "lon"]].head(10).to_string(index=False))

print(f"Records with longitude outside [{LON_MIN}–{LON_MAX}]: {len(bad_lon):,}")
if len(bad_lon) > 0:
    print(bad_lon[["il", "ilce", "mahalle", "lat", "lon"]].head(10).to_string(index=False))

coord_per_dist = tr_df.dropna(subset=["lat", "lon"]).groupby("ilce").apply(
    lambda g: g[["lat", "lon"]].drop_duplicates().shape[0]
)
single_coord_dists = coord_per_dist[coord_per_dist == 1]
print(f"\nDistricts with ALL records at a SINGLE coordinate point: {len(single_coord_dists):,}")
print("  (These districts likely have district-level centroid rather than per-neighbourhood coords.)")
if len(single_coord_dists) > 0:
    print(f"  Sample: {list(single_coord_dists.index[:10])}")

multi_coord_dists = coord_per_dist[coord_per_dist > 1]
print(f"Districts with multiple distinct coordinate points: {len(multi_coord_dists):,}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(coord_per_dist, bins=50, color="#e74c3c", edgecolor="white", alpha=0.85)
ax.set_title("Unique Coordinate Points per District", fontsize=13)
ax.set_xlabel("# Distinct (lat, lon) pairs")
ax.set_ylabel("# Districts")
plt.tight_layout()
plt.savefig("figures/fig12_coord_per_district.png")
plt.close()
print("[Saved] figures/fig12_coord_per_district.png")

df["mahalle_stripped"] = df["mahalle"].astype(str).str.strip()
trailing_space = (df["mahalle"] != df["mahalle_stripped"]).sum()
very_short = (df["mahalle"].astype(str).str.len() <= 2).sum()
print(f"\nNeighbourhood names with trailing/leading spaces : {trailing_space:,}")
print(f"Neighbourhood names ≤ 2 characters              : {very_short:,}")
if very_short > 0:
    print(df[df["mahalle"].astype(str).str.len() <= 2][["il", "ilce", "mahalle"]].to_string(index=False))

df.drop(columns=["mahalle_stripped"], inplace=True)

print("\n" + "=" * 70)
print("9. KEY FINDINGS SUMMARY")
print("=" * 70)

total_rows = len(df)
unique_il = df["il"].nunique()
unique_ilce = df["ilce"].nunique()
unique_semt = df["semt"].nunique()
unique_mah = df["mahalle"].nunique()
unique_pk = df["PK"].nunique()
missing_lat_pct = df["lat"].isnull().mean() * 100
missing_lon_pct = df["lon"].isnull().mean() * 100
max_share = mah_per_pk.max()
max_share_pk = mah_per_pk.idxmax()

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│                     KEY FINDINGS                                 │
├─────────────────────────────────────────────────────────────────┤
│  Total records            : {total_rows:>8,}                          │
│  Unique provinces (il)    : {unique_il:>8,}  (81 TR + 1 KKTC)        │
│  Unique districts (ilce)  : {unique_ilce:>8,}                          │
│  Unique townships (semt)  : {unique_semt:>8,}                          │
│  Unique neighbourhoods    : {unique_mah:>8,}                          │
│  Unique postal codes (PK) : {unique_pk:>8,}                          │
├─────────────────────────────────────────────────────────────────┤
│  Missing Latitude         : {missing_lat_pct:>7.2f}%  (all from KKTC)     │
│  Missing Longitude        : {missing_lon_pct:>7.2f}%  (all from KKTC)     │
│  Full-row duplicates      : {full_dups:>8,}                          │
├─────────────────────────────────────────────────────────────────┤
│  Max neighbourhoods per PK: {max_share:>8,}  (PK={max_share_pk})              │
│  Districts w/ 1 coord pt  : {len(single_coord_dists):>8,}  (centroid coords)   │
│  Out-of-bbox coordinates  : {len(bad_lat)+len(bad_lon):>8,}                          │
└─────────────────────────────────────────────────────────────────┘
""")

print("NOTABLE INSIGHTS:")
print("  1. KKTC records (Northern Cyprus) have NO latitude/longitude — all")
print("     coordinate fields are empty, making spatial analysis impossible")
print("     for KKTC entries.")
print()
print("  2. Many TR districts share a SINGLE coordinate point across all")
print(f"     their neighbourhoods ({len(single_coord_dists):,} districts). This indicates the")
print("     dataset uses district-level centroids, not per-neighbourhood GPS.")
print()
print("  3. Postal codes are heavily shared — up to", max_share,
      f"neighbourhoods share PK {max_share_pk}.")
print("     The postal system is designed at district/township level,")
print("     not per-neighbourhood.")
print()
print("  4. Latitude is positively correlated with longitude (Turkey's")
print("     coastline runs NW–SE), but both are weakly correlated with")
print("     postal code numbers (PK is assigned administratively, not")
print("     geographically).")
print()
top_il_name = df["il"].value_counts().idxmax()
top_il_cnt  = df["il"].value_counts().max()
print(f"  5. {top_il_name} has the most records ({top_il_cnt:,}), which is counter-intuitive")
print("     since ISTANBUL and ANKARA are Turkey's largest cities. Black Sea")
print("     and eastern provinces top the list because they have many small")
print("     rural villages each listed as individual neighbourhood entries.")

print("\n" + "=" * 70)
print("EDA complete. All figures saved in ./figures/")
print("=" * 70)

print("\n" + "=" * 70)
print("10. POSTAL CODE → PROVINCE VALIDATION")
print("=" * 70)

PLATE_TO_IL = {
    "01":"ADANA","02":"ADIYAMAN","03":"AFYONKARAHISAR","04":"AGRI",
    "05":"AMASYA","06":"ANKARA","07":"ANTALYA","08":"ARTVIN",
    "09":"AYDIN","10":"BALIKESIR","11":"BILECIK","12":"BINGOL",
    "13":"BITLIS","14":"BOLU","15":"BURDUR","16":"BURSA",
    "17":"CANAKKALE","18":"CANKIRI","19":"CORUM","20":"DENIZLI",
    "21":"DIYARBAKIR","22":"EDIRNE","23":"ELAZIG","24":"ERZINCAN",
    "25":"ERZURUM","26":"ESKISEHIR","27":"GAZIANTEP","28":"GIRESUN",
    "29":"GUMUSHANE","30":"HAKKARI","31":"HATAY","32":"ISPARTA",
    "33":"MERSIN","34":"ISTANBUL","35":"IZMIR","36":"KARS",
    "37":"KASTAMONU","38":"KAYSERI","39":"KIRKLARELI","40":"KIRSEHIR",
    "41":"KOCAELI","42":"KONYA","43":"KUTAHYA","44":"MALATYA",
    "45":"MANISA","46":"KAHRAMANMARAS","47":"MARDIN","48":"MUGLA",
    "49":"MUS","50":"NEVSEHIR","51":"NIGDE","52":"ORDU",
    "53":"RIZE","54":"SAKARYA","55":"SAMSUN","56":"SIIRT",
    "57":"SINOP","58":"SIVAS","59":"TEKIRDAG","60":"TOKAT",
    "61":"TRABZON","62":"TUNCELI","63":"SANLIURFA","64":"USAK",
    "65":"VAN","66":"YOZGAT","67":"ZONGULDAK","68":"AKSARAY",
    "69":"BAYBURT","70":"KARAMAN","71":"KIRIKKALE","72":"BATMAN",
    "73":"SIRNAK","74":"BARTIN","75":"ARDAHAN","76":"IGDIR",
    "77":"YALOVA","78":"KARABUK","79":"KILIS","80":"OSMANIYE",
    "81":"DUZCE",
}

tr_only = df[df["il"] != "KKTC"].copy()
tr_only["PK_padded"] = tr_only["PK"].astype(str).str.zfill(5)
tr_only["pk_prefix"] = tr_only["PK_padded"].str[:2]
tr_only["expected_il"] = tr_only["pk_prefix"].map(PLATE_TO_IL)

mismatch = tr_only[tr_only["expected_il"] != tr_only["il"]]
match_rate = (1 - len(mismatch) / len(tr_only)) * 100

print(f"\n  Total TR records checked : {len(tr_only):,}")
print(f"  Matching (PK prefix == il): {len(tr_only) - len(mismatch):,}  ({match_rate:.2f}%)")
print(f"  Mismatches               : {len(mismatch):,}  ({100 - match_rate:.2f}%)")

if len(mismatch) > 0:
    print("\n  Sample mismatches (il in file vs expected from PK prefix):")
    cols_show = ["il", "ilce", "mahalle", "PK", "PK_padded", "pk_prefix", "expected_il"]
    print(mismatch[cols_show].head(15).to_string(index=False))

    mis_by_il = mismatch.groupby("il").size().sort_values(ascending=False)
    print(f"\n  Mismatches by province (top 10):")
    print(mis_by_il.head(10).to_string())

    fig, ax = plt.subplots(figsize=(12, 5))
    top_mis = mis_by_il.head(20)
    ax.bar(top_mis.index, top_mis.values, color="#e74c3c", edgecolor="white")
    ax.set_title("Postal Code Prefix–Province Mismatches by Province (Top 20)", fontsize=13)
    ax.set_xlabel("Province (İl)")
    ax.set_ylabel("# Mismatched Records")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("figures/fig13_pk_province_mismatch.png")
    plt.close()
    print("\n[Saved] figures/fig13_pk_province_mismatch.png")
else:
    print("\n  No mismatches found — all postal code prefixes match province names.")

print("\n" + "=" * 70)
print("11. NEIGHBOURHOOD TYPE CLASSIFICATION")
print("=" * 70)

SUFFIX_MAP = {
    "MAH":    "Mahalle (Neighbourhood)",
    "KOYU":   "Köy (Village)",
    "KOY":    "Köy (Village)",
    "BELDE":  "Belde (Township)",
    "MEZRA":  "Mezra (Hamlet)",
    "KASABA": "Kasaba (Town)",
    "BUCAK":  "Bucak (Sub-district)",
    "KÖYÜ":   "Köy (Village)",       # with ö — just in case
}

def classify_suffix(name):
    name = str(name).strip().upper()
    for suffix, label in SUFFIX_MAP.items():
        if name.endswith(suffix):
            return label
    return "Other / Unknown"

df["mah_type"] = df["mahalle"].apply(classify_suffix)

type_counts = df["mah_type"].value_counts()
print("\nNeighbourhood type distribution (all records):")
for t, cnt in type_counts.items():
    pct = cnt / len(df) * 100
    print(f"  {t:<35}: {cnt:>7,}  ({pct:.2f}%)")

top20_il = df["il"].value_counts().head(20).index.tolist()
sub_top20 = df[df["il"].isin(top20_il)]
pivot = (sub_top20.groupby(["il", "mah_type"])
         .size()
         .unstack(fill_value=0)
         .loc[top20_il])

col_order = type_counts.index.tolist()
col_order = [c for c in col_order if c in pivot.columns]
pivot = pivot[col_order]

colors_type = sns.color_palette("Set2", len(col_order))
fig, ax = plt.subplots(figsize=(14, 6))
pivot.plot(kind="bar", stacked=True, ax=ax, color=colors_type, edgecolor="white", width=0.75)
ax.set_title("Neighbourhood Type Breakdown – Top 20 Provinces", fontsize=13)
ax.set_xlabel("Province (İl)")
ax.set_ylabel("Record Count")
ax.legend(title="Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("figures/fig14_mah_type_by_province.png")
plt.close()
print("\n[Saved] figures/fig14_mah_type_by_province.png")

fig, ax = plt.subplots(figsize=(8, 6))
wedge_colors = sns.color_palette("Set2", len(type_counts))
ax.pie(type_counts.values, labels=type_counts.index,
       autopct="%1.1f%%", colors=wedge_colors, startangle=140,
       pctdistance=0.82, labeldistance=1.05)
ax.set_title("Overall Neighbourhood Type Distribution", fontsize=13)
plt.tight_layout()
plt.savefig("figures/fig15_mah_type_pie.png")
plt.close()
print("[Saved] figures/fig15_mah_type_pie.png")

df["is_urban"] = df["mah_type"] == "Mahalle (Neighbourhood)"
urban_ratio = (df.groupby("il")["is_urban"].mean() * 100).sort_values(ascending=False)
print("\nMost urban provinces (% Mahalle records):")
print(urban_ratio.head(10).round(1).to_string())
print("\nMost rural provinces (% non-Mahalle records):")
print(urban_ratio.tail(10).sort_values().round(1).to_string())

fig, ax = plt.subplots(figsize=(14, 5))
colors_ur = ["#3498db" if v >= 50 else "#e67e22" for v in urban_ratio.values]
ax.bar(urban_ratio.index, urban_ratio.values, color=colors_ur, edgecolor="white")
ax.axhline(50, color="red", linestyle="--", linewidth=1, label="50% threshold")
ax.set_title("Urban Ratio by Province (% Mahalle records) – blue=majority urban", fontsize=12)
ax.set_xlabel("Province")
ax.set_ylabel("% Mahalle (urban)")
ax.legend()
plt.xticks(rotation=90, fontsize=6)
plt.tight_layout()
plt.savefig("figures/fig16_urban_ratio_by_province.png")
plt.close()
print("[Saved] figures/fig16_urban_ratio_by_province.png")

print("\n" + "=" * 70)
print("12. DUPLICATE ROOT-CAUSE ANALYSIS")
print("=" * 70)

dup_mask = df.duplicated(subset=["il", "ilce", "semt", "mahalle"], keep=False)
dup_df = df[dup_mask].copy()

dups_by_il = dup_df.groupby("il").size().sort_values(ascending=False)
print(f"\n  Total records involved in duplicates: {len(dup_df):,}")
print(f"  Unique provinces affected           : {dup_df['il'].nunique()}")
print(f"\n  Duplicate records by province (top 15):")
print(dups_by_il.head(15).to_string())

dups_by_ilce = dup_df.groupby(["il", "ilce"]).size().sort_values(ascending=False)
print(f"\n  Duplicate records by district (top 10):")
print(dups_by_ilce.head(10).to_string())

dup_exact = df.duplicated(keep=False)  # full 7-column match
dup_loc_only = dup_mask & ~df.duplicated(keep=False)  # location match but coord differs
print(f"\n  Full exact duplicates (all 7 cols match)          : {df.duplicated().sum():,}")
print(f"  Location-only dups (coords differ between copies) : {dup_loc_only.sum():,}")

fig, ax = plt.subplots(figsize=(12, 5))
top_dup = dups_by_il.head(20)
ax.bar(top_dup.index, top_dup.values, color="#e74c3c", edgecolor="white")
ax.set_title("Duplicate Records by Province (Top 20)", fontsize=13)
ax.set_xlabel("Province (İl)")
ax.set_ylabel("# Records in Duplicates")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("figures/fig17_duplicates_by_province.png")
plt.close()
print("\n[Saved] figures/fig17_duplicates_by_province.png")

print("\n" + "=" * 70)
print("13. GEOGRAPHIC DENSITY HEATMAP")
print("=" * 70)

coord_tr = tr_df.dropna(subset=["lat", "lon"])

fig, ax = plt.subplots(figsize=(12, 7))
hb = ax.hexbin(coord_tr["lon"], coord_tr["lat"],
               gridsize=80, cmap="YlOrRd", mincnt=1, linewidths=0.2)
cb = fig.colorbar(hb, ax=ax, label="Record Count")
ax.set_title("Geographic Density of Administrative Records – Turkey (Hexbin)", fontsize=13)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_xlim(25, 45)
ax.set_ylim(35, 43)
plt.tight_layout()
plt.savefig("figures/fig18_hexbin_density.png", dpi=150)
plt.close()
print("[Saved] figures/fig18_hexbin_density.png")

sample_kde = coord_tr.sample(min(20000, len(coord_tr)), random_state=42)
fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(sample_kde["lon"], sample_kde["lat"],
           s=0.5, alpha=0.15, color="#95a5a6", zorder=1)
sns.kdeplot(data=sample_kde, x="lon", y="lat",
            levels=8, fill=True, cmap="Blues", alpha=0.6, ax=ax, zorder=2)
sns.kdeplot(data=sample_kde, x="lon", y="lat",
            levels=8, fill=False, cmap="Blues", linewidths=0.8, ax=ax, zorder=3)
ax.set_title("Geographic Density – KDE Contour (Turkey)", fontsize=13)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_xlim(25, 45)
ax.set_ylim(35, 43)
plt.tight_layout()
plt.savefig("figures/fig19_kde_density.png", dpi=150)
plt.close()
print("[Saved] figures/fig19_kde_density.png")

print("\n" + "=" * 70)
print("14. PROVINCE-LEVEL SUMMARY TABLE")
print("=" * 70)

summary = df.groupby("il").agg(
    total_records   = ("mahalle",  "count"),
    unique_districts= ("ilce",     "nunique"),
    unique_townships= ("semt",     "nunique"),
    unique_mahalle  = ("mahalle",  "nunique"),
    unique_pk       = ("PK",       "nunique"),
    lat_min         = ("lat",      "min"),
    lat_max         = ("lat",      "max"),
    lon_min         = ("lon",      "min"),
    lon_max         = ("lon",      "max"),
    pct_mahalle     = ("is_urban", "mean"),
).reset_index()

summary["lat_range"] = (summary["lat_max"] - summary["lat_min"]).round(4)
summary["lon_range"] = (summary["lon_max"] - summary["lon_min"]).round(4)
summary["pct_mahalle"] = (summary["pct_mahalle"] * 100).round(1)
summary = summary.sort_values("total_records", ascending=False).reset_index(drop=True)

print(f"\n  Province summary table ({len(summary)} rows):")
print(summary[["il","total_records","unique_districts","unique_townships",
               "unique_mahalle","unique_pk","pct_mahalle"]].to_string(index=False))

summary.to_csv("province_summary.csv", index=False)
print("\n[Saved] province_summary.csv")

print("\n" + "=" * 70)
print("15. POSTAL CODE GAP ANALYSIS")
print("=" * 70)

tr_pk_padded = tr_only["PK_padded"].dropna().unique()
tr_pk_int = sorted([int(p) for p in tr_pk_padded if p.isdigit()])

present_codes = set(f"{int(p)//1000:02d}" for p in tr_pk_padded if p.isdigit())
expected_codes = set(f"{i:02d}" for i in range(1, 82))
missing_codes  = expected_codes - present_codes
extra_codes    = present_codes - expected_codes   # should be empty for TR

print(f"\n  Expected province codes (01–81): {len(expected_codes)}")
print(f"  Present in dataset             : {len(present_codes)}")
print(f"  Missing province codes         : {sorted(missing_codes) if missing_codes else 'None'}")
print(f"  Unexpected codes (>81)         : {sorted(extra_codes) if extra_codes else 'None'}")

pk_range_by_prov = (tr_only.groupby("pk_prefix")["PK_padded"]
                    .agg(lambda x: sorted(x.unique())))
pk_stats = tr_only.groupby("pk_prefix")["PK_padded"].agg(
    pk_count="nunique",
    pk_min=lambda x: x.min(),
    pk_max=lambda x: x.max(),
).reset_index().sort_values("pk_prefix")
pk_stats["province"] = pk_stats["pk_prefix"].map(PLATE_TO_IL)
print(f"\n  Postal codes per province (first 20):")
print(pk_stats.head(20)[["pk_prefix","province","pk_count","pk_min","pk_max"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(pk_stats["province"], pk_stats["pk_count"],
       color=sns.color_palette("tab20", len(pk_stats)), edgecolor="white")
ax.set_title("Number of Unique Postal Codes per Province", fontsize=13)
ax.set_xlabel("Province")
ax.set_ylabel("# Unique PKs")
plt.xticks(rotation=90, fontsize=6)
plt.tight_layout()
plt.savefig("figures/fig20_pk_count_per_province.png")
plt.close()
print("\n[Saved] figures/fig20_pk_count_per_province.png")

fig, ax = plt.subplots(figsize=(13, 8))
for _, row in pk_stats.iterrows():
    prov_idx = int(row["pk_prefix"])
    pks_in_prov = tr_only[tr_only["pk_prefix"] == row["pk_prefix"]]["PK_padded"].unique()
    pk_ints = [int(p) for p in pks_in_prov if p.isdigit()]
    ax.scatter(pk_ints, [prov_idx] * len(pk_ints), s=4, alpha=0.6, color="#2980b9")
ax.set_title("Postal Code Number Space Coverage by Province", fontsize=13)
ax.set_xlabel("Postal Code (5-digit)")
ax.set_ylabel("Province Code (plate number)")
ax.set_yticks(range(1, 82))
ax.set_yticklabels([f"{i:02d}" for i in range(1, 82)], fontsize=5)
plt.tight_layout()
plt.savefig("figures/fig21_pk_space_coverage.png", dpi=150)
plt.close()
print("[Saved] figures/fig21_pk_space_coverage.png")

print("\n" + "=" * 70)
print("16. WORD CLOUD OF MAHALLE NAMES")
print("=" * 70)

try:
    from wordcloud import WordCloud

    STOP = {"MAH", "KOYU", "KOY", "BELDE", "MEZRA", "KASABA",
            "BUCAK", "MAHALLESI", "KOYU", "KÖYÜ", "MAHALLESİ"}

    import re
    cleaned = (df["mahalle"].astype(str).str.upper()
               .str.replace(r"\(.*?\)", "", regex=True)   # remove (...) blocks
               .str.replace(r"[^A-Z\s]", "", regex=True)  # keep letters & spaces
               .str.strip())
    all_words = " ".join(cleaned.tolist())
    tokens = [w for w in all_words.split() if w not in STOP and len(w) > 2]
    word_freq = pd.Series(tokens).value_counts()
    print(f"\n  Top 20 most frequent words in mahalle names:")
    print(word_freq.head(20).to_string())

    wc = WordCloud(
        width=1400, height=700,
        background_color="white",
        colormap="tab20",
        max_words=200,
        collocations=False,
    ).generate_from_frequencies(word_freq.to_dict())

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most Frequent Words in Neighbourhood Names", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig("figures/fig22_mahalle_wordcloud.png", dpi=150)
    plt.close()
    print("\n[Saved] figures/fig22_mahalle_wordcloud.png")

    fig, ax = plt.subplots(figsize=(14, 5))
    top_words = word_freq.head(30)
    ax.bar(top_words.index, top_words.values,
           color=sns.color_palette("tab20", 30), edgecolor="white")
    ax.set_title("Top 30 Most Frequent Content Words in Neighbourhood Names", fontsize=13)
    ax.set_xlabel("Word")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("figures/fig23_mahalle_top_words.png")
    plt.close()
    print("[Saved] figures/fig23_mahalle_top_words.png")

except ImportError:
    print("  [SKIP] wordcloud package not installed. Run: pip install wordcloud")

print("\n" + "=" * 70)
print("17. COORDINATE PRECISION ANALYSIS")
print("=" * 70)

def decimal_places(series):
    """Count decimal digits for each float value in a Series."""
    return (series.dropna()
            .astype(str)
            .str.split(".")
            .str[1]
            .str.len()
            .fillna(0)
            .astype(int))

lat_prec = decimal_places(coord_tr["lat"])
lon_prec = decimal_places(coord_tr["lon"])

print("\nLatitude decimal-place distribution:")
print(lat_prec.value_counts().sort_index().to_string())
print("\nLongitude decimal-place distribution:")
print(lon_prec.value_counts().sort_index().to_string())

coord_tr_cp = coord_tr.copy()
coord_tr_cp["coord_key"] = coord_tr_cp["lat"].astype(str) + "," + coord_tr_cp["lon"].astype(str)
shared_coord = coord_tr_cp.groupby("ilce")["coord_key"].transform("nunique") == 1
pct_shared = shared_coord.mean() * 100
print(f"\n  Records sitting on a district-shared centroid: {shared_coord.sum():,}  ({pct_shared:.1f}%)")
print(f"  Records with unique-within-district coords   : {(~shared_coord).sum():,}  ({100-pct_shared:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(lat_prec.value_counts().sort_index().index,
            lat_prec.value_counts().sort_index().values,
            color="#3498db", edgecolor="white")
axes[0].set_title("Latitude – Decimal Places")
axes[0].set_xlabel("Decimal places")
axes[0].set_ylabel("# Records")

axes[1].bar(lon_prec.value_counts().sort_index().index,
            lon_prec.value_counts().sort_index().values,
            color="#e67e22", edgecolor="white")
axes[1].set_title("Longitude – Decimal Places")
axes[1].set_xlabel("Decimal places")
axes[1].set_ylabel("# Records")

plt.suptitle("Coordinate Precision: Number of Decimal Places in Lat/Lon", fontsize=13)
plt.tight_layout()
plt.savefig("figures/fig24_coord_precision.png")
plt.close()
print("\n[Saved] figures/fig24_coord_precision.png")

sample_prec = coord_tr_cp.sample(min(20000, len(coord_tr_cp)), random_state=42)
shared_sample = coord_tr_cp.loc[sample_prec.index, "coord_key"].map(
    coord_tr_cp["coord_key"].value_counts()
)
colors_prec = ["#e74c3c" if v > 1 else "#2ecc71" for v in shared_sample.values]
fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(sample_prec["lon"], sample_prec["lat"],
           s=1.5, alpha=0.5, c=colors_prec)
from matplotlib.lines import Line2D
legend_elements = [Line2D([0],[0], marker="o", color="w", markerfacecolor="#e74c3c",
                          markersize=8, label="Shared centroid coord"),
                   Line2D([0],[0], marker="o", color="w", markerfacecolor="#2ecc71",
                          markersize=8, label="Unique coord")]
ax.legend(handles=legend_elements, loc="upper left")
ax.set_title("Records with Shared Centroid vs Unique Coordinates", fontsize=13)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_xlim(25, 45)
ax.set_ylim(35, 43)
plt.tight_layout()
plt.savefig("figures/fig25_shared_vs_unique_coords.png", dpi=150)
plt.close()
print("[Saved] figures/fig25_shared_vs_unique_coords.png")

print("\n" + "=" * 70)
print("ALL SECTIONS COMPLETE. Figures saved in ./figures/")
print("=" * 70)
