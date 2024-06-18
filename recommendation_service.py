import pandas as pd

class RecommendationService:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def get_recommendations(self, skin_type, issue):
        # Filter products based on skin type and issue
        filtered_products = self.df[
            (self.df[skin_type] == 'x') & 
            (self.df[issue] == 'x')
        ]
        
        # Select relevant columns to return as recommendation
        recommendations = filtered_products[['ID', 'name', 'label', 'brand', 'price', 'rank', 'ingredients']].to_dict(orient='records')
        
        return recommendations

# Initialize the recommendation service with the CSV file path
recommendation_service = RecommendationService('Product.csv')

def get_recommendations(skin_type, issue):
    return recommendation_service.get_recommendations(skin_type, issue)
