import pandas as pd

class RecommendationService:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, encoding='ISO-8859-1')

    def get_recommendations(self, skin_type, issue):
        recommendations = self.df[(self.df[skin_type] == 'Yes') & (self.df[issue] == 'Yes')]
        return recommendations['ID'].tolist()

recommendation_service = RecommendationService('Product.csv')

def get_recommendations(skin_type, issue):
    return recommendation_service.get_recommendations(skin_type, issue)
