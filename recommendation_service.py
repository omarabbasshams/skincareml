import pandas as pd

class RecommendationService:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, encoding='latin1')
        
    def get_recommendations(self, answers, issue):
        # Assuming issue is a string that matches one of the columns ['acne', 'redness', 'wrinkles', 'bags']
        issue_recommendations = self.df[self.df[issue] == 1]
        
        # Filter further based on answers if needed
        for question_id, answer in answers.items():
            issue_recommendations = issue_recommendations[issue_recommendations[str(answer)] == 1]
        
        recommendations = issue_recommendations[['ID', 'Label', 'brand', 'name', 'price', 'rank', 'ingredients']].to_dict('records')
        return recommendations
