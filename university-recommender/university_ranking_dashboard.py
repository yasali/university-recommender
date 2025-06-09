import pandas as pd
import gradio as gr
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = "top-university-rankings.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Initial data exploration
print(df.head())

# Example: Clustering universities based on selected features
features = df[['Academic_Reputation_Score', 'Employer_Reputation_Score', 'Citations_per_Faculty_Score']]
kmeans = KMeans(n_clusters=5, random_state=0).fit(features)
df['Cluster'] = kmeans.labels_

# Visualization: Bar chart of top universities by overall score
fig = px.bar(df.sort_values('Overall_Score', ascending=False).head(10),
             x='Institution_Name', y='Overall_Score',
             title='Top 10 Universities by Overall Score')

# Handle missing values by imputing with the mean
X = df[['Academic_Reputation_Score', 'Employer_Reputation_Score', 'Citations_per_Faculty_Score']]
y = df['Overall_Score']

# Convert data to numeric, coercing errors to NaN
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Re-impute missing values by imputing with the mean
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Prepare data for predictive modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict and evaluate
predictions = regressor.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Recommendation system based on user preferences
def recommend_universities(location=None, focus=None, size=None):
    filtered_df = df
    if location:
        filtered_df = filtered_df[filtered_df['Location'] == location]
    if focus:
        filtered_df = filtered_df[filtered_df['FOCUS'] == focus]
    if size:
        filtered_df = filtered_df[filtered_df['SIZE'] == size]
    return filtered_df[['Institution_Name', 'Overall_Score']].sort_values(by='Overall_Score', ascending=False).head(10)

# Update Gradio dashboard
def show_dashboard():
    with gr.Blocks() as dashboard:
        gr.Markdown("# 🌍 QS World University Rankings 2025 Dashboard")
        
        # Top Universities Chart
        gr.Markdown("## Top Universities by Overall Score")
        gr.Plot(fig, label="Top Universities")
        
        # Predictive Model Performance
        gr.Markdown("## Predictive Modeling Performance")
        mse_text = gr.Textbox(label="Mean Squared Error", value=str(mse), interactive=False)
        
        # University Recommendations
        gr.Markdown("## University Recommendations")
        with gr.Row():
            location_input = gr.Textbox(label="Preferred Location")
            focus_input = gr.Textbox(label="Preferred Focus")
            size_input = gr.Textbox(label="Preferred Size")
        recommend_btn = gr.Button("Recommend")
        recommendations_output = gr.Dataframe(headers=['Institution Name', 'Overall Score'])
        recommend_btn.click(fn=recommend_universities, inputs=[location_input, focus_input, size_input], outputs=[recommendations_output])
    return dashboard

if __name__ == "__main__":
    dashboard = show_dashboard()
    dashboard.launch() 