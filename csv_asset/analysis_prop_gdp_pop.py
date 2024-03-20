import pandas as pd

# Load the CountryInfo_with_Population_GDP_Score_Updated.csv to analyze the data
df = pd.read_csv('/home/work/climate-cooperation-competition/new_csv/CountryInfo_with_All_Ranks.csv')

# Filter out countries with Rank_Score = 1 or 2
filtered_countries = df[df["Rank_pop"].isin([1])]

# Calculate the total GDP and population of these countries
total_gdp_filtered = filtered_countries["GDP"].sum()
total_population_filtered = filtered_countries["Population"].sum()

# Calculate the total world GDP and population from the dataset
total_world_gdp = df["GDP"].sum()
total_world_population = df["Population"].sum()

# Calculate the percentage of total GDP and population occupied by Rank_Score = 1 countries
gdp_percentage = (total_gdp_filtered / total_world_gdp) * 100
population_percentage = (total_population_filtered / total_world_population) * 100

print({"total_gdp_filtered": total_gdp_filtered, "total_world_gdp": total_world_gdp, "gdp_percentage": gdp_percentage, "total_population_filtered": total_population_filtered, "total_world_population": total_world_population, "population_percentage": population_percentage})