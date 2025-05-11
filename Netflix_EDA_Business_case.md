#Business Goal

Netflix aims to optimize its content investment by understanding what type of content to prioritize, which genres and regions perform best, when content should be released, and which directors or actors drive the most engagement. This data-driven strategy should help Netflix enhance user engagement, increase viewership, and maintain a competitive edge in global markets.
##Objectives

1. Determine whether Netflix should invest more in TV Shows or Movies.

2. Identify the most successful genres across different regions.

3. Determine optimal release periods for content.

4. Identify the most productive and recurring directors and actors.

5. Understand regional content production trends.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
##1. Data Overview & Cleaning

**Dataset**: Netflix's global content catalog (TV Shows and Movies)

**Rows**: 8807

**Columns**: 12 (title, type, director, cast, country, date_added, release_year, rating, duration, listed_in, description)

**Preprocessing** **Steps**:

1. Converted 'date_added' to datetime format.


2. Split 'duration' into numerical columns: 'movie_minutes' for Movies and 'show_seasons' for TV Shows.

3. Unnested multiple values in 'director', 'cast', and 'country' columns.

4. Dropped missing or malformed values when necessary.
df=pd.read_csv('Netflix_analysis.csv')
#shape of the Actual data
df.shape
df.head(5)
## Data Cleaning
###Unnesting of columns : Directors, Casts,Listed_in
New_unnesting changes

# Create a copy and drop rows where 'cast' is missing (to safely split
# and explode)
df_exploded = df.dropna(subset=['cast']).copy()

# Split comma-separated values into lists
df_exploded['cast'] = df_exploded['cast'].str.split(', ')
df_exploded['listed_in'] = df_exploded['listed_in'].str.split(', ')
df_exploded['director'] = df_exploded['director'].str.split(',')

# Explode both 'cast' and 'listed_in' into individual rows
df_exploded = df_exploded.explode('cast')
df_exploded = df_exploded.explode('listed_in')
df_exploded = df_exploded.explode('director')

# Reset index for clean output
df_exploded = df_exploded.reset_index(drop=True)

# Show the resulting DataFrame
df_exploded

director = (
    pd.DataFrame(df["director"].apply(lambda x: str(x).split(", ")).tolist(), index=df["title"])
    .stack()
    .reset_index(level=1, drop=True)
    .reset_index()
    .rename(columns={0: "Director"})
)


cast = (
    pd.DataFrame(df["cast"].apply(lambda x: str(x).split(", ")).tolist(), index=df["title"])
    .stack()
    .reset_index(level=1, drop=True)
    .reset_index()
    .rename(columns={0: "Cast"})
)

listed_in = (
    pd.DataFrame(df["listed_in"].apply(lambda x: str(x).split(", ")).tolist(), index=df["title"])
    .stack()
    .reset_index(level=1, drop=True)
    .reset_index()
)
#Unique Values
df.nunique()
director
cast
listed_in
#Changing Datatype
df["date_added"] = pd.to_datetime(df["date_added"], errors = "coerce")
df
### Duration
# Extract number of seasons (e.g., "2 Seasons" → 2)
df.loc[df["type"] == "TV Show", "show_seasons"] = (
    df.loc[df["type"] == "TV Show", "duration"]
    .dropna()
    .apply(lambda x: int(x.split()[0]))
)
df

# Extract movie duration in minutes (e.g., "90 min" → 90)
df.loc[df["type"] == "Movie", "movie_minutes"] = (
    df.loc[df["type"] == "Movie", "duration"]
    .dropna()
    .apply(lambda x: int(x.split()[0]))
)
df

####Handling Null Values
New changes
# Make a copy of the exploded DataFrame for cleaning
df_cleaned = df_exploded.copy()

# a. Handle nulls in categorical columns by replacing with "Unknown <Column Name>"
categorical_cols = ['cast', 'director', 'country', 'date_added', 'rating', 'duration', 'listed_in']
for col in categorical_cols:
    df_cleaned[col] = df_cleaned[col].fillna(f'Unknown {col.title()}')

# b. Handle nulls in continuous (numeric) columns by replacing with 0
continuous_cols = ['release_year']
for col in continuous_cols:
    df_cleaned[col] = df_cleaned[col].fillna(0)

# Show cleaned DataFrame
df_cleaned

df_cleaned.info()
# prompt: find the unique ID's
#  df_cleaned.nunique()

df_cleaned.nunique()

 Describe the cleaned csv file
# prompt:  df_cleaned.describe()
#  df.describe(include = object)

df_cleaned.describe()
df_cleaned.describe(include = object)

### Merging Dataframes for directors,cast and Listed_in
# Merging director information into the main dataframe
director_df = df.merge(director, on="title", how="left")

# Merging cast information
cast_df = df.merge(cast, on="title", how="left")

# Merging listed_in (categories/genres) information
listed_in_df = df.merge(listed_in, on="title", how="left")

director_df
cast_df
listed_in_df
#Data Cleaning
df['date_added'].unique()
#Non-Geographical Analysis
# Value counts for different columns, excluding ratings with count 1
rating_counts = df['rating'].value_counts()
filtered_rating_counts = rating_counts[rating_counts > 1]
print(filtered_rating_counts)

print(df['type'].value_counts())
print(df['country'].value_counts())

# Unique attributes for specified columns
print("Unique Directors:", director['Director'].nunique())
print("Unique Cast Members:", cast['Cast'].nunique())
print("Unique Categories:", listed_in[0].nunique())
#Visual Analysis


# Univariate Analysis
plt.figure(figsize=(10, 6))
sns.countplot(x='type', data=df)
plt.title('Distribution of Content Types')
plt.show()

# Calculate rating counts and filter out those with count <= 1
rating_counts = df['rating'].value_counts()
filtered_ratings = rating_counts[rating_counts > 1]

# Use the filtered ratings to create the countplot
plt.figure(figsize=(12, 6))
sns.countplot(y='rating',
              data=df[df['rating'].isin(filtered_ratings.index)],  # Filter data
              order=filtered_ratings.index)  # Order by filtered ratings
plt.title('Distribution of Content Ratings (Excluding Ratings with Count <= 1)')
plt.show()
plt.figure(figsize=(10, 6))
df['country'].value_counts().nlargest(10).plot(kind='bar') # top 10 countries
plt.title('Top 10 Countries with Content on Netflix')
plt.xlabel('Country')
plt.ylabel('Number of Titles')
plt.show()


#Movie duration distribution
plt.figure(figsize=(10,6))
sns.histplot(df[df['type'] == 'Movie']['movie_minutes'], bins=30, kde=True)
plt.title('Distribution of Movie Durations')
plt.xlabel('Movie Duration (minutes)')
plt.ylabel('Frequency')
plt.show()
#TV Show Season Distribution
plt.figure(figsize=(10,6))
sns.histplot(df[df['type'] == 'TV Show']['show_seasons'], bins=30, kde=True)
plt.title('Distribution of TV Show Seasons')
plt.xlabel('Number of Seasons')
plt.ylabel('Frequency')
plt.show()

📊 Insights from TV Show Season Distribution
🎬 One-and-Done Shows Rule

The majority of TV shows have only 1 season (~1800+).

This suggests a trend of limited series or shows that got canceled early.

➡️ Mini-series and short-term content are very common.

📉 Sharp Decline After Season 2

Season 2 shows drop to under 500.

There’s a steep fall in shows with 3, 4, or more seasons.

🚫 Fewer renewals = cautious platform investment.

🏆 Long Runners Are Rare Gems

Only a small number of shows make it past 5 seasons.

📺 10+ season shows are outliers, likely major hits or legacy content.

🧭 Overall Content Strategy?

Heavy skew toward short-form, experimental, or test-run content.

Platforms may prioritize quantity over long-term storytelling.

🔁 Binge-worthy, quick-consumption shows dominate.
# prompt: explore more bivariate relationships (e.g., country vs rating, director vs type)  also replace nan values with "Unknown" .i want top 10 of them

import matplotlib.pyplot as plt
# Bivariate Analysis: Country vs Rating (Top 10 Countries)
top_10_countries = df['country'].value_counts().nlargest(10).index
plt.figure(figsize=(12, 6))
sns.countplot(x='country', hue='rating', data=df[df['country'].isin(top_10_countries)], order=top_10_countries)
plt.title('Content Rating Distribution by Country (Top 10)')
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
plt.show()


# Bivariate Analysis: Director vs Type (Top 10 Directors)
# First, fill NaN values in 'director' column with "Unknown" (already done in your code)
# Then, find the top 10 directors
top_10_directors = director_df['Director'].value_counts().nlargest(10).index
plt.figure(figsize=(12, 6))
sns.countplot(x='Director', hue='type', data=director_df[director_df['Director'].isin(top_10_directors)], order=top_10_directors)
plt.title('Content Type Distribution by Director (Top 10)')
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
plt.show()

🎞️ Top Named Directors

All other visible directors in the top 10 have contributed almost exclusively to Movies, not TV Shows:

Chilaka appears to be the most prolific named director.

Directors like Martin Scorsese, Raúl Campos, and Wael Chawki have modest movie contributions.

📺 No named director among the top 10 shows significant involvement in TV Shows, further highlighting the NaN issue.

📌 Key Takeaways

🧹 Cleaning or imputing director names (wherever possible) is crucial for meaningful insights.

🎬 Named directors are mostly involved in Movies, not TV Shows.

📉 The dominance of missing data severely reduces the utility of this visualization in its current state.
plt.figure(figsize=(10, 6))
sns.boxplot(x='type', y='movie_minutes', data=df)  # Use movie_minutes for movies
plt.title('Movie Duration by Content Type')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='type', y='show_seasons', data=df)  # Use show_seasons for TV shows
plt.title('Number of Seasons by Content Type')
plt.show()
🎬 Movies

 ⏱️ Median duration is around 95–100 minutes.

 📈 Most movies range between 50 to 150 minutes.

 ⚠️ Numerous outliers above 150 minutes, going as high as 310+ mins — these could be extended editions, documentaries, or rare long-format films.

📉 Some short films (<30 mins) also exist, seen on the lower end.

📺 TV Shows

⌛ The duration is essentially 0 minutes, indicating:

🧩 Possibly missing or zero-filled data for episode runtime.

🛠️ A need to clean or impute episode duration for proper analysis.
top_countries = df['country'].value_counts().nlargest(5).index  # Get the top 5 countries
plt.figure(figsize=(10, 6))
sns.countplot(data=df[df['country'].isin(top_countries)], x='country', hue='rating')
plt.title('Distribution of ratings across top 5 countries')
plt.show()
🌍 Top 5 Countries by Content Ratings
🇺🇸 United States

🔥 TV-MA (Mature Audience) dominates with the highest count – over 900!

👶 Also high in TV-Y7 and TV-14, showing a wide range of age-targeted content.

🎯 Broad diversity in ratings suggests a mature, diverse content library.

❓ Unknown

🟧 TV-MA is also the most common, followed by TV-14 and TV-PG.

👻 Lack of country data, but resembles U.S. rating trends.

🇮🇳 India

🥇 TV-14 is heavily dominant – nearly 550 entries!

🔞 Mature ratings like TV-MA appear in fewer numbers.

🎭 Suggests more family-friendly or teen-rated TV content.

🇬🇧 United Kingdom

📺 TV-MA still leads but with a smaller count.

🧒 Mix of TV-PG, TV-14, and TV-Y7 visible.

🇬🇧 Offers a balanced content mix but less skewed toward mature content.

🇯🇵 Japan

🟠 Primarily has TV-MA and TV-14, with some TV-PG.

🧠 Possibly anime or drama-heavy, with moderate mature content presence.


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# 1. Get Top 10 Countries
top_10_countries = df['country'].value_counts().nlargest(10).index
filtered_df = df[df['country'].isin(top_10_countries)]

# 2. One-Hot Encoding for Categorical Features
encoded_df = pd.get_dummies(filtered_df, columns=['country', 'rating'], prefix=['country', 'rating'])

# 3. Function to Calculate Cramér's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

# 4. Create Correlation Matrix using Cramér's V
# Select features for analysis: 'show_seasons', encoded country, and rating columns
features = ['show_seasons'] + [col for col in encoded_df.columns if 'country_' in col or 'rating_' in col]
analysis_df = encoded_df[features]

# Initialize an empty matrix to store Cramér's V values
corr_matrix = pd.DataFrame(index=['show_seasons'], columns=analysis_df.columns[1:])

# Calculate Cramér's V and fill the correlation matrix
for col in analysis_df.columns[1:]:
    v = cramers_v(analysis_df['show_seasons'], analysis_df[col])
    corr_matrix.loc['show_seasons', col] = v

# 5. Create Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix.astype(float), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Association between Season Count, Rating, and Top 10 Countries (Cramér\'s V)')
plt.xlabel('Country and Rating (One-Hot Encoded)')
plt.ylabel('Show Seasons')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# Investigate the relationship between country and the average movie duration.
country_movie_duration = df.groupby('country')['movie_minutes'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='country', y='movie_minutes', data=country_movie_duration.sort_values('movie_minutes', ascending=False).head(10))
plt.title('Average Movie Duration by Country (Top 10)')
plt.xlabel('Country')
plt.ylabel('Average Movie Duration (minutes)')
plt.xticks(rotation=45, ha='right')
plt.show()
🌍 Top 10 Countries by Average Movie Duration

🥇 United States, Japan (~210 min)
  Movies produced jointly by the US and Japan have the longest average duration.
 🕰️ Likely due to epic-style films, documentaries, or anime collaborations.

🥈 United Kingdom, Morocco (~205 min)

  Surprisingly long movies from this pairing!

 🎥 May reflect historical dramas or co-productions with extended storytelling.

🥉 Liechtenstein (~200 min)
 A small nation, but apparently makes long-form films.
 🎞️ Possibly due to a few long-duration entries skewing the average.

🇺🇸 United States (~190+ min)
 Classic Hollywood movies are often long-format, especially blockbusters or
  director’s cuts.

🇩🇪 Germany (~188 min)
 Known for historical and art films, which can run long.

🇧🇪 🇪🇸 Belgium, Spain (~180 min)
 European co-productions likely emphasize deep narrative or period dramas.

🇭🇰 🇸🇬 Hong Kong, Singapore (~170 min)
 Lengthy action thrillers or drama-packed cinema may contribute here.

🏴‍☠️ Soviet Union (~165 min)
 Soviet-era films were often philosophical or ideological epics.

🇺🇸 🇲🇦 United States, Morocco (~163 min)
 Possibly travel-based or war-related collaborations with longer runtimes.

🇬🇧 United Kingdom (~162 min)
 UK films, especially period pieces or biopics, tend to be long and detailed.
import matplotlib.pyplot as plt
# Calculate the average number of seasons per year
df['year_added'] = df['date_added'].dt.year
seasons_per_year = df.groupby('year_added')['show_seasons'].mean()

# Create the plot
plt.figure(figsize=(10, 6))
seasons_per_year.plot(kind='line', marker='o')
plt.title('Average Number of Seasons per Year')
plt.xlabel('Year Added')
plt.ylabel('Average Number of Seasons')
plt.grid(True)
plt.show()

🔍 Insights from "Average Number of Seasons per Year"

📈 Big Spike in 2013
  2013 had the highest average number of seasons per show ➡️ above 1.0!
  👉 Lots of multi-season shows added this year!

❌📺 Dry Spell (2009–2012)
  These years show an average of 0 seasons –
  💤 Possibly only short content, one-season shows, or no series added at all.

⬆️ Comeback After 2018
  After a dip in 2018, there's a steady upward trend through 2021
  🌱 More content with multiple seasons started appearing again!

🎢 Fluctuating Phase (2014–2018)
  The average seasons per year went up and down
  🤷 Strategy may have shifted frequently or viewer demand varied.

🕰️ Decent Start in 2008
  The year 2008 had a moderate average (~0.5)
  ✅ Indicates some multi-season content right at the beginning.

📊 Stable Baseline from 2015 Onwards
  Average never dropped below ~0.35 post-2015
  🧱 Shows a consistent content strategy with some longevity.
# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()
📊 Interpretation

 This negative correlation does not imply causation, but rather a structural separation in the dataset:

 A content item is either a movie (with duration in minutes) or a TV show (with number of seasons).

 Netflix has likely encoded both content types in the same dataset, leading to this logical separation.

💡 Business/Analytical Implications

 This correlation confirms clean and mutually exclusive classification of content types.

 It suggests that using both features together without filtering could confuse predictive models—you should split data or handle missing/inapplicable values for each type.

 In clustering or recommendation systems, you should treat movie_minutes and show_seasons as domain-specific to movies and shows respectively.
# Pairplot for numerical features
sns.pairplot(df[numerical_features])
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()
🔍 Univariate Insights (Diagonal Histograms)

1. Movie Duration (movie_minutes)

  Most movies are between 60–120 minutes.

 There's a right-skewed distribution—a few movies are unusually long (up to 300+ minutes).

 A spike around the short duration (~0–10 min) likely represents missing values or short films (e.g., trailers or shorts).

2. Show Seasons (show_seasons)
 The vast majority of shows have 1 season.

 Very few shows go beyond 3–4 seasons, with rare cases up to 15+ seasons.

 This suggests Netflix focuses more on limited series or newer shows.

🔄 Bivariate Insights (Scatterplots)
 3. No Overlap Between Movies and Shows
 Movies have non-zero movie_minutes and zero show_seasons.

 TV shows have non-zero show_seasons and zero movie_minutes.

 There’s a clear separation—no content exists that has both movie length and multiple seasons, which aligns with how Netflix classifies content.

 4. Sparse Extremes
 A few outliers exist:

 Very long movies (~300 mins).

 Shows with 10+ seasons—probably long-running series acquired from traditional networks (e.g., Friends, The Office).

🧠 Business Implications

 Content Categorization is clean and binary—Netflix ensures distinct separation between movies and TV shows in metadata.

 Netflix’s show catalog skews toward newer, shorter series, which are easier to produce and consume.

 Long-form or legacy content is rare and exceptional, possibly due to licensing limits or strategic focus on binge-able content.


import matplotlib.pyplot as plt
# Group data by year and type, then count the number of movies/shows
yearly_content = df.groupby([df['date_added'].dt.year, 'type']).size().reset_index(name='count')

# Separate data for movies and TV shows
movies = yearly_content[yearly_content['type'] == 'Movie']
tv_shows = yearly_content[yearly_content['type'] == 'TV Show']

# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(movies['date_added'], movies['count'], label='Movies', marker='o')
plt.plot(tv_shows['date_added'], tv_shows['count'], label='TV Shows', marker='o')

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Number of Movies/TV Shows Added')
plt.title('Distribution of Movies and TV Shows Added on Netflix per Year')
plt.legend()
plt.grid(True)
plt.show()

🧠 Strategic Business Insights
Shift in Focus Around 2015:

Netflix significantly ramped up original content production starting in 2015, with aggressive content acquisitions following shortly after.

The growth supports Netflix’s expansion into international markets and subscription-driven engagement.

Movies Lead in Quantity, but Shows Catch Up in Stability:

Movies have always outnumbered shows, but the TV show addition trend is smoother, indicating a more stable content pipeline.

TV shows may have a longer viewer lifecycle due to episodes and seasons, making them strategic for retention.

Pandemic Impact:

Both content types saw post-2019 drops, emphasizing the industry-wide effect of the pandemic.

Netflix's ability to maintain substantial additions even during disruption speaks to its resilient infrastructure.
import matplotlib.pyplot as plt
# Group the data by release year and count the number of movies released each year
release_year_counts = df.groupby('release_year')['title'].count()

# Filter for the last 30 years
last_30_years = release_year_counts.tail(30)

# Create the plot
plt.figure(figsize=(12, 6))
last_30_years.plot(kind='bar')
plt.title('Number of Movies Released per Year (Last 30 Years)')
plt.xlabel('Release Year')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

💡 Business Insights
Platform Maturity:

The 2015–2019 boom aligns with Netflix’s transformation into a full-scale content producer, not just a distributor.

Pandemic Effect:

The decline post-2019 is significant and reflects the global industry slowdown, yet Netflix maintained a relatively high volume compared to pre-2016.

Recommendation:

Compare post-2021 trends with viewer engagement to assess if fewer, better-quality releases result in higher ROI.

Leverage historical growth trends to forecast future release volume under varying production conditions.
### Missing Value & Outlier Check

import matplotlib.pyplot as plt
# Check for missing values in the entire DataFrame
print(df.isnull().sum())

# Check for outliers using boxplots for numerical features
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x=df['movie_minutes'])
plt.title('Boxplot of Movie Minutes')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['show_seasons'])
plt.title('Boxplot of Show Seasons')

plt.tight_layout()
plt.show()


🎬 Boxplot: Movie Minutes

Median Duration: Most movies have a median duration around 90–100 minutes, which aligns with standard feature-length films.

Interquartile Range (IQR): Majority of movies fall between approximately 60 to 120 minutes.

Outliers:

There are several extreme outliers above 250 minutes, possibly including extended cuts or documentaries.

Short-duration outliers below ~30 mins may be short films or mislabeled content.

📺 Boxplot: Show Seasons

Median Season Count: Most TV shows have just 1 season, indicating a large number of limited series or short-run shows.

Interquartile Range (IQR): Ranges from about 1 to 2 seasons.

Outliers:

Many shows go beyond 3–5 seasons, with a few reaching up to 17 seasons, showing long-running series (e.g., classic sitcoms or anime).

These long-runners are rare and skew the data distribution.
import matplotlib.pyplot as plt

# Group the data by release year and count the number of movies released each year
release_year_counts = df.groupby('release_year')['title'].count()

# Filter for the last 30 years (adjust as needed)
last_30_years = release_year_counts.tail(30)

# Create the plot using a line plot instead of a bar plot
plt.figure(figsize=(12, 6))
last_30_years.plot(kind='line', marker='o', linestyle='-')  # Use 'line' plot type
plt.title('Number of Movies Released per Year (Last 30 Years)')
plt.xlabel('Release Year')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.grid(True)  # Add grid for better readability
plt.tight_layout()
plt.show()

📊 Insights from Movie Release Trends (Last 30 Years)


📈 Steady Growth (1990s–2014):

From ~1990 to 2014, movie releases grew gradually.

The count remained under 400 per year until 2014.

This reflects the traditional pace of production and limited global reach before the streaming boom.

🚀 Rapid Surge (2015–2019):

2015–2019 shows an explosive increase, peaking around 2018–2019 with over 1,100+ movies per year.

This spike correlates with:

Netflix's aggressive content investment, especially international expansion.

Streaming becoming mainstream, increasing both original and licensed content on the platform.

📉 Sharp Decline Post-2020:

A noticeable drop in 2020–2021, falling to below 600 titles.

Likely caused by:

COVID-19 pandemic impact (halted productions, distribution delays).

Shift toward TV shows and mini-series.

Content consolidation due to increasing competition and cost control.


 # total movies/TV shows in each genre
 x = listed_in_df.groupby(['listed_in' , 'type'])['show_id'].count().reset_index()
 x.pivot(index = 'listed_in' , columns = 'type' , values = 'show_id').sort_index()
 x
🔍 Insights

🎬 Multi-Genre Tagging
Many titles are listed under multiple genres (e.g., "Action & Adventure, Anime Features"), indicating Netflix's strategy to maximize discoverability through cross-category indexing.

🔄 Repetition of Genre Themes
Repeated categories like “Action & Adventure”, “Anime Features”, “TV Horror”, and “Thrillers” suggest strong user interest in these segments.

📊 Higher Volume of Movies
A quick scan shows a slightly higher concentration of "Movie" entries than TV Shows, consistent with Netflix’s legacy as a movie-streaming platform.

🧒 Youth & Niche Targeting
Genres like “Teen TV Shows,” “Children & Family Movies,” and “TV Mysteries” point to a deliberate effort to cater to younger demographics and niche interests.


#top 10 actors in the world

# Assuming 'cast' is a column in your DataFrame and contains comma-separated actor names.
# If not, replace 'cast' with the correct column name.

top_actors = cast_df['Cast'].value_counts().nlargest(10)
top_actors

🔍 Insights on Cast Appearance Frequency

⚠️ High Missing Values

825 entries are missing cast information (NaN).

This is a significant data gap and may affect accuracy of actor-based trend analysis.

🎭 Top Frequent Actors
The most featured actors are:

Anupam Kher (43 titles)

Shah Rukh Khan (35 titles)

Julie Tejwani (33)

Naseeruddin Shah (32)

Takahiro Sakurai (32)
This suggests:

Strong presence of Indian actors, confirming India’s heavy content contribution.

Takahiro Sakurai and Yuki Kaji signal a solid amount of Japanese anime or voiceover content.

🌍 Global Mix of Talent
While Indian actors dominate the top, there’s a visible international mix—a hint of Netflix's global content strategy (e.g., Japan).


# Filter movies from the last 5 years
current_year = pd.Timestamp.now().year
recent_movies = df[(df['release_year'] >= current_year - 5) & (df['type'] == 'Movie')]

# Group by genre and sum viewership (replace 'viewership' with your actual metric)
# If you don't have a viewership column, use a proxy like the count of titles in each genre.
most_viewed_genres = recent_movies.groupby('listed_in')['title'].count().reset_index()

# Rename columns for clarity
most_viewed_genres.rename(columns = {'listed_in': 'Genre', 'title':'Number of Movies'}, inplace=True)


# Sort by viewership (or your chosen metric) in descending order and get top 10
most_viewed_genres = most_viewed_genres.sort_values(by='Number of Movies', ascending=False)


# Display the most viewed genres
print(most_viewed_genres.head(10))


🎬 Insights: Genre Frequency in Netflix Movies

✅ Top Genres by Frequency:

Stand-Up Comedy leads with 53 movies, showing strong demand for light, short-form content.

Dramas + International Movies (combo) also perform well with 52 movies, highlighting global storytelling appeal.

Children & Family Movies (46) and Documentaries (43) also have a strong presence, suggesting a balanced portfolio across age groups and interests.

🎭 Cross-Genre Combinations Are Common:

Many high-frequency genres are combinations like:

Comedies, Dramas, International Movies (35)

Dramas, International Movies, Romantic Movies (31)

Dramas, International Movies, Thrillers (24)

This indicates Netflix often categorizes content under multiple overlapping genres to improve discoverability and viewer relevance.

🌍 Strong Focus on International Appeal:

Nearly all top genre combinations include "International Movies".

Reinforces Netflix’s global-first content strategy—tailoring stories for international audiences.



#1. Identify the top 5 countries
top_5_countries = df['country'].value_counts().nlargest(5).index

#2. Filter the dataframe for those top 5 countries
top_countries_df = df[df['country'].isin(top_5_countries)]

#3. Explode the listed_in column to handle multiple genres per movie/show
exploded_df = top_countries_df.assign(listed_in=top_countries_df['listed_in'].str.split(', ')).explode('listed_in')

#4. Group by country and listed_in to count the number of titles
genre_counts_by_country = exploded_df.groupby(['country', 'listed_in']).size().reset_index(name='counts')

#5. Sort and select the top 10 genres in each country
top_genres_by_country = genre_counts_by_country.groupby('country').apply(lambda x: x.nlargest(10, 'counts')).reset_index(drop=True)


#6. Plot using a bar plot with seaborn
plt.figure(figsize=(15, 8))
sns.barplot(x='listed_in', y='counts', hue='country', data=top_genres_by_country)
plt.xlabel('Genre')
plt.ylabel('Number of Titles')
plt.title('Top 10 Genres across Top 5 Countries')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

### Insights
**Genre Insights by Country**

**India**

Top genres: International Movies, Dramas, Comedies

Strong preference for movies over shows

High interest in Romantic and Music & Musicals content

**United States**

Top genres: Dramas, Comedies, Docuseries, Documentaries

Balanced demand for Movies and TV Shows

High traction in factual and crime-related content

Japan

Dominated by Anime Series, Anime Features, Horror Movies

Strong niche preference for animation and horror

United Kingdom

Leading in Documentaries, British TV, TV Comedies

Emphasis on TV Shows and factual content

Unknown Category

Mix of International Movies, TV Dramas, Docuseries, Family Movies

Likely globally licensed or untagged content
###Actionable Recommendations

📺 Expand TV Show Production in the US & UK
Prioritize genres like Docuseries, Crime, and TV Comedies to meet growing demand.

🎬 Continue Investing in Movies for India
Focus on Dramas, Romantic, and International Movies, which dominate Indian viewership.

🎨 Boost Anime Content for Japan
Scale up acquisition and production of Anime Series and Anime Features to maintain cultural relevance.

🌍 Tailor Content per Region
Use genre preferences to inform regional marketing and production (e.g., Music & Musicals in India, British TV in the UK).

🗓️ Align Genre Drops with Regional Trends
Schedule content releases by season and cultural context to maximize visibility (e.g., horror around Halloween in Japan/US).

🧩 Clarify & Tag 'Unknown' Data
Investigate and clean the “Unknown” country segment to better understand global genre performance.
# Final Business Insights
1️⃣ Content Duration & Structure
🎬 Movie Duration by Content Type (Box Plot)
⏱️ Movies: Median ~95–100 mins, with outliers >310 mins (e.g. epics or docs).

⚠️ TV Shows: Duration mostly 0 → ⛔ likely due to missing episode length data.

📏 Average Movie Duration by Country (Top 10)
🥇 US-Japan 🇺🇸🇯🇵: Longest films (~210 mins) – likely collabs or anime epics.

🇬🇧🇲🇦 UK-Morocco: ~205 mins, could be historical/war dramas.

🇱🇮 Liechtenstein: ~200 mins, possibly skewed by few long films.

🎭 Art-house and co-productions dominate longer runtimes.

2️⃣ Content Strategy Over Time
⏳ Average Number of Seasons Per Year
🔥 2013 spike: Highest avg. number of seasons.

❌ 2009–2012: Almost no multi-season content.

🌱 Post-2018: Gradual increase = renewed long-form strategy.

🎢 2014–2018: Fluctuations = inconsistent planning.

🧱 Post-2015: Stable baseline (avg. > 0.35 seasons/year).

3️⃣ TV Show Characteristics
📺 Distribution of TV Show Seasons
1️⃣ Single-season shows dominate (~1800+): Reflects trend in mini-series or cancellations.

📉 Steep drop after 2 seasons – very few shows go beyond 5.

🏆 10+ season shows = rare legacy hits.

4️⃣ Content Ratings by Country
🔢 Distribution of Ratings Across Top 5 Countries
🇺🇸 US: TV-MA leads (~900+), but wide range (TV-Y7, TV-14).

🇮🇳 India: TV-14 dominant → family/teen focus.

🇬🇧 UK: Balanced mix but skewed toward moderate maturity.

🇯🇵 Japan: Mostly TV-MA & TV-14 → anime/drama-driven.

❓ Unknown: Mimics U.S. rating profile → likely metadata gaps.

5️⃣ Director Information & Attribution
🎬 Content Type Distribution by Director (Top 10)
❗ Missing Director Info (NaN): Dominates dataset (~2400+ TV Shows).

📽️ Named directors (e.g. Scorsese, Chilaka) appear almost exclusively for movies.

📉 Skew limits the value of director-based insights unless cleaned.

6️⃣ Numerical Feature Correlation
🔁 Correlation Matrix of movie_minutes & show_seasons
➖ Negative Correlation (-0.61): Movies have durations, shows have seasons – mutually exclusive.

✅ Confirms clean separation between content types.

⚠️ Treat these features independently in models or filters.

🧠 Overall Takeaways
Netflix content is:

🎯 Strategically short-term (1-season shows, compact films).

🌍 Diverse across countries and production styles.

🔍 Dominated by mature-rated and director-less metadata.

📊 Structured for separation between TV and movies—ideal for filtered analysis.
#Recommendations

1. Enhance Global Talent Representation

Action: Broaden talent sourcing to ensure a diverse mix of global actors, especially for regions like Japan and non-English speaking countries.

Reason: The presence of actors like Takahiro Sakurai (Japanese) and Shah Rukh Khan (Indian) highlights Netflix's international content strategy. By diversifying the cast further, Netflix can expand its global appeal and strengthen regional content resonance.

2. Address Missing Cast Data

Action: Prioritize cleaning up the missing cast information (825 entries with NaN values) to improve data accuracy and actor-based trend analysis.

Reason: Missing data can skew insights, affecting recommendations for talent engagement and casting decisions. Ensuring that every title is properly tagged will improve accuracy in assessing cast popularity.

3. Focus on High-Frequency Actors

Action: Continue leveraging top actors like Anupam Kher and Shah Rukh Khan, as their frequent presence in titles indicates strong audience interest.

Reason: The success of popular actors across multiple titles can drive better viewership and retention, suggesting that these actors have a proven track record of drawing in audiences.

4. Expand Content for Niche Markets

Action: Consider more content with actors like Takahiro Sakurai, who signify the popularity of anime in Japan.

Reason: By identifying regional preferences (e.g., anime in Japan), Netflix can continue tailoring its content offering for niche markets, potentially creating loyalty and community engagement.

5. Strengthen Regional Content Production

Action: Continue building out strong local content, particularly focusing on regions with high actor frequency (India and Japan, for example).

Reason: Regional content tends to perform well when local talent is involved, and Netflix has seen success with Bollywood actors and Japanese voice talents.

6. Improve Genre-based Content Discovery

Action: Implement a more structured approach to genre tagging (splitting the multi-genre tags for better tracking).

Reason: More precise genre classification will help in better content discovery for users and provide more insightful trend analysis, ultimately guiding content strategies.

7. Global Strategy with Local Relevance

Action: Use insights from top actors and genres to strengthen global-local content strategies, ensuring content resonates with diverse audiences worldwide while staying true to local tastes.

Reason: Netflix’s ability to successfully tailor its content (like anime for Japan and Bollywood for India) speaks to the value of global-local strategies.

  8.Content Strategy Improvements
Prioritize Data Quality:

Fix TV Show Duration: Impute or clean missing episode duration data to improve the accuracy of TV Show duration analysis.

Director Attribution: Clean or fill missing director information (NaN) to ensure more meaningful insights can be extracted from director-based trends, especially in movies.

Leverage Short-Term Content:

Focus on Mini-Series: Given the dominance of 1-season shows, platforms may continue or increase investment in short-form, high-impact content. This also aligns with binge-watching behaviors.

9. Content Duration Strategy
Optimize Film Length:

With long-duration films (outliers > 150 mins), try offering extended or special edition versions that attract niche audiences.

Target shorter runtimes (90–120 minutes) for broader appeal, optimizing viewer retention.

Fix TV Show Duration:

Impute missing duration or episode data for better analysis and visibility into show engagement.

If unavailable, label “0 minutes” episodes as missing to avoid skewed conclusions.

10.TV Show & Movie Dynamics
Optimize TV Show Formats:

Given the steep drop after 1-2 seasons, consider creating more episodic mini-series or anthology shows that require less commitment.

For long-running shows (10+ seasons), these should be treated as flagship content that can sustain long-term engagement and viewer loyalty.

Explore Collaborations & Co-productions:

Since certain regions (like Japan, UK, and the US) produce long-format films, look into collaborative international projects that blend different cultural perspectives and appeal to wider audiences.

11 Content Rating Optimization
Adapt Ratings for Regional Preferences:

India: Embrace the TV-14 strategy, focusing on family-friendly or teen-oriented content.

U.S. & Japan: Focus on TV-MA content for mature audiences, especially for high-visual drama or anime content.

Metadata Review: Correct the “Unknown” ratings for better regional tailoring and ensure metadata is up-to-date for more precise content categorization.

12.User Experience Enhancements
Viewer Retention Analysis:

Shorter seasons (1–2) may correlate with higher churn rates. To keep viewers engaged, platforms can experiment with episodic cliffhangers or seasonal arcs that encourage return viewing.

Recommendation Engine Improvements:

Content Personalization: Use the insights about movie length, season duration, and ratings to refine recommendation algorithms that cater to specific user preferences for film length, genre, or episode count.

13.Further Data Insights
Cross-Platform Insights:

Investigate platform-specific trends, as this data can reveal how different platforms manage season lengths and content durations. This can help inform decisions about where to allocate resources and production effort.

Content Lifecycle Analysis:

Evaluate the lifetime of popular shows or films based on their ratings and seasons. This could predict potential cancellations or renewals, guiding future content investment decisions.
