# ==========================================================
# NETFLIX DATA ANALYSIS PROJECT
# Author: Syed Muhammad Abdul Nafeah
# Tools: Python, Pandas
# ==========================================================

# -----------------------------
# Import Libraries
# -----------------------------

import pandas as pd
import numpy as np

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("datasets/netflix_titles.csv")

print("="*60)
print("NETFLIX DATA ANALYSIS PROJECT")
print("="*60)

# -----------------------------
# Dataset Overview
# -----------------------------

print("\nDataset Shape")
print(df.shape)

print("\nRows :", df.shape[0])
print("Columns :", df.shape[1])

print("\nColumn Names")
print(df.columns.tolist())

print("\nFirst Five Rows")
print(df.head())

print("\nLast Five Rows")
print(df.tail())

# -----------------------------
# Dataset Information
# -----------------------------

print("\nDataset Info")
print(df.info())

print("\nData Types")
print(df.dtypes)

print("\nSummary Statistics")
print(df.describe(include='all'))

# -----------------------------
# Missing Values
# -----------------------------

print("\nMissing Values")
print(df.isnull().sum())

# Remove duplicate records

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")

# -----------------------------
# Total Missing Values
# -----------------------------

print("\nTotal Missing Values")
print(df.isnull().sum().sum())

# ==========================================================
# DATA CLEANING
# ==========================================================

print("\nCleaning Dataset...")


df["director"] = df["director"].fillna("Unknown")
df["cast"] = df["cast"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna("Not Rated")
df = df.dropna(subset=["date_added"])
df = df.dropna(subset=["duration"])
df["date_added"] = pd.to_datetime(
    df["date_added"].str.strip(),
    format="%B %d, %Y"
)
text_columns = [
    "title",
    "director",
    "cast",
    "country",
    "rating",
    "listed_in"
]

for column in text_columns:
    df[column] = df[column].str.strip()

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

print("\nCreating New Features...")

df["year_added"] = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month_name()
df["day_added"] = df["date_added"].dt.day

movie_duration = (
    df["duration"]
    .str.extract(r"(\d+)")
    .astype(float)
)
df["duration_number"] = movie_duration

# Display New Columns

print("\nNew Columns Added")
print(df.head())

print("\nCurrent Shape")
print(df.shape)

# Save cleaned dataset
df.to_csv("datasets/cleaned_netflix.csv", index=False)

print("\nCleaned dataset saved successfully!")


# ==========================================================
# PART 2 - EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

print("\n" + "="*60)
print("PART 2 - EXPLORATORY DATA ANALYSIS")
print("="*60)


print("\n1. Movies vs TV Shows")
print(df["type"].value_counts())


print("\n2. Percentage")

type_percent = round(df["type"].value_counts(normalize=True) * 100, 2)
print(type_percent)


print("\n3. Top 10 Countries")

countries = (
    df["country"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

print(countries)


print("\n4. Top 10 Directors")

directors = (
    df[df["director"] != "Unknown"]["director"]
    .value_counts()
    .head(10)
)

print(directors)


print("\n5. Top 10 Actors")

actors = (
    df[df["cast"] != "Unknown"]["cast"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

print(actors)


print("\n6. Rating Distribution")

print(df["rating"].value_counts())


print("\n7. Top Genres")

genres = (
    df["listed_in"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

print(genres)

print("\n8. Top Release Years")

release = (
    df["release_year"]
    .value_counts()
    .sort_index()
)

print(release.tail(20))


print("\n9. Content Added to Netflix")

added = (
    df["year_added"]
    .value_counts()
    .sort_index()
)

print(added)


print("\n10. Uploads By Month")

print(df["month_added"].value_counts())

print("\n11. Oldest Title")

oldest = df[df["release_year"] == df["release_year"].min()]

print(oldest[["title","release_year","type"]])

print("\n12. Newest Title")

newest = df[df["release_year"] == df["release_year"].max()]

print(newest[["title","release_year","type"]])

movies = df[df["type"]=="Movie"]

avg_duration = np.mean(movies["duration_number"])

print("\n13. Average Movie Duration")
print(round(avg_duration,2),"minutes")

longest = movies.loc[movies["duration_number"].idxmax()]

print("\n14. Longest Movie")

print(longest[["title","duration"]])


shortest = movies.loc[movies["duration_number"].idxmin()]

print("\n15. Shortest Movie")

print(shortest[["title","duration"]])

duration = movies["duration_number"].dropna().to_numpy()

print("\n16. NumPy Statistics")

print("Mean   :", np.mean(duration))
print("Median :", np.median(duration))
print("Maximum:", np.max(duration))
print("Minimum:", np.min(duration))
print("Std Dev:", round(np.std(duration),2))


above_avg = movies[movies["duration_number"] > np.mean(duration)]

print("\n17. Movies Longer Than Average")

print("Total:", len(above_avg))

shows = df[df["type"]=="TV Show"]

print("\n18. TV Shows")

print("Total TV Shows:",len(shows))

india = df[df["country"].str.contains("India",case=False,na=False)]

print("\n19. Indian Content")

print(india[["title","type","release_year"]].head(20))

print("\nTotal Indian Titles:",len(india))

print("\n20. First 20 Titles Alphabetically")

print(df.sort_values("title")[["title","type"]].head(20))

# SAVE RESULTS

print("\nSaving files...")

# Save cleaned dataset
df.to_csv("datasets/cleaned_netflix.csv", index=False)

# Save top countries
countries.to_csv("datasets/top_countries.csv")

# Save top directors
directors.to_csv("datasets/top_directors.csv")

# Save top actors
actors.to_csv("datasets/top_actors.csv")

# Save top genres
genres.to_csv("datasets/top_genres.csv")

# Save ratings
df["rating"].value_counts().to_csv("datasets/ratings.csv")

print("Files saved successfully!")

# ==========================================
# PROJECT SUMMARY
# ==========================================

print("\n========== PROJECT SUMMARY ==========")
print("Original Shape :", df.shape)
print("Movies         :", len(movies))
print("TV Shows       :", len(shows))
print("Top Country    :", countries.index[0])
print("Top Director   :", directors.index[0])
print("Top Genre      :", genres.index[0])
print("Average Duration :", round(avg_duration,2),"minutes")
