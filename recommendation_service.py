import pandas as pd

class RecommendationService:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        print(f"DataFrame Loaded: {self.df.head()}")  # Log DataFrame head for verification

    def get_recommendations(self, skin_type, issue):
        print(f"Filtering for skin_type: {skin_type} and issue: {issue}")  # Log input criteria
        # Update the filter logic to check for '1' instead of 'Yes'
        recommendations = self.df[(self.df[skin_type] == 1) & (self.df[issue] == 1)]
        print(f"Filtered Recommendations: {recommendations}")  # Log filtered data
        return recommendations['ID'].tolist()

recommendation_service = RecommendationService('Product.csv')

def get_recommendations(skin_type, issue):
    # Log a sample of the DataFrame
    print(f"Sample DataFrame: {recommendation_service.df.sample(5)}")
    return recommendation_service.get_recommendations(skin_type, issue)
