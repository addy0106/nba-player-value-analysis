# NBA Player Value Analysis - 2023/24 Season

## Project Overview

This project analyzes NBA player performance and contract data to identify undervalued and overvalued players using web scraping, exploratory data analysis, and machine learning.

The core idea mirrors what NBA front offices actually do: find players who deliver outsized performance relative to their salary. Data was scraped live from Basketball Reference using Python, cleaned and merged, then analyzed using a custom Undervalued Player Score and a Random Forest machine learning model.

## Tools and Technologies

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Random Forest Regressor)
- BeautifulSoup, Requests (Web Scraping)
- Data Source: Basketball Reference (basketball-reference.com)

## Project Structure

-> resume_project_4/
->data/ per_game_stats.csv , salaries.csv               # Scraped player contract data


->visuals/top15_undervalued_players.png, 
top10_overvalued_players.png, 
salary_vs_points.png, 
performance_by_position.png, 
feature_importance.png, 
actual_vs_predicted_salary.png

->scraper.py                     # Web scraping script for stats and salaries
->analysis.py                    # EDA, undervalued scoring, and ML model
->README.md

## Methodology

### Step 1 - Data Collection
Player per game statistics and salary contract data were scraped directly from Basketball Reference using the Requests and BeautifulSoup libraries. Data was cleaned, deduplicated, and merged on player name. Players with fewer than 20 games played or less than 15 minutes per game were excluded to ensure statistical reliability.

### Step 2 - Undervalued Player Score
A composite performance score was created using the following formula:

Performance Score = PTS + (0.75 x AST) + (0.75 x TRB) + STL + BLK - (0.5 x TOV)

Both performance and salary were normalized to a 0 to 1 scale. The Undervalued Score was then calculated as:

Undervalued Score = Normalized Performance Score - Normalized Salary

A higher score indicates a player delivers strong production relative to their contract value. A lower score indicates a player is being paid more than their stats justify.

### Step 3 - Machine Learning Model
A Random Forest Regressor was trained to predict player salary from on-court performance statistics.

Features used: Age, Games Played, Minutes Per Game, Points, Assists, Rebounds, Steals, Blocks, Turnovers, Field Goal Percentage, Three Point Percentage, Free Throw Percentage, Effective Field Goal Percentage, Performance Score, Position

Train/Test Split: 80/20
Mean Absolute Error: $6.42M
R-squared Score: 0.673

The model explains approximately 67% of the variation in NBA player salaries using only performance statistics, which highlights how much salary is also influenced by factors outside of pure production such as market size, contract timing, and age.

## Key Findings

1. Victor Wembanyama ranked as the most undervalued player in the league, posting 21.4 points, 10.6 rebounds, and 3.9 assists per game on a rookie contract of $13.4M.

2. Cam Thomas, Jalen Williams, and Paolo Banchero also ranked highly as undervalued, all producing at a high level on early career contracts.

3. Points per game and overall performance score were the strongest predictors of salary according to feature importance analysis.

4. Centers and Power Forwards showed the highest variance in performance scores across positions, reflecting the wide range of roles big men play in the modern NBA.

5. The gap between actual and predicted salary for several veterans suggests that past reputation and contract timing play a significant role in NBA compensation beyond current performance.

## Visualizations

The following charts were generated and saved to the visuals/ folder:

- Top 15 Most Undervalued NBA Players
- Top 10 Most Overvalued NBA Players
- Salary vs Points Per Game (colored by Undervalued Score)
- Performance Score Distribution by Position
- Random Forest Feature Importance for Salary Prediction
- Actual vs Predicted Player Salary

## How to Run

Step 1 - Install dependencies:
pip install requests beautifulsoup4 lxml pandas numpy matplotlib seaborn scikit-learn

Step 2 - Scrape data from Basketball Reference:
python scraper.py

Step 3 - Run analysis and generate all visuals:
python analysis.py

All output charts will be saved to the visuals/ folder automatically.

## Author

Aditya Dixit
Arizona State University - B.S. Data Science
https://linkedin.com/in/aditya-dixit-36b649273
https://github.com/addy0106
