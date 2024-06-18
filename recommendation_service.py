import pandas as pd

class RecommendationService:
    def __init__(self, csv_path):
        try:
            self.df = pd.read_csv(csv_path, encoding='ISO-8859-1')  # Specify the encoding
        except UnicodeDecodeError:
            raise ValueError("The CSV file is not properly encoded. Please ensure it is saved with the correct encoding.")

    def get_recommendations(self, skin_type, issue):
        # Filter the DataFrame based on the skin type and issue
        filtered_df = self.df[(self.df[skin_type] == 'Yes') & (self.df[issue] == 'Yes')]
        # Return the recommended product names
        recommendations = filtered_df['name'].tolist()
        return recommendations
