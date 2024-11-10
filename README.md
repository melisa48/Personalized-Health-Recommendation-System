# Personalized Health Recommendations System
- This Personalized Health Recommendation System is an AI-driven application that provides tailored health recommendations based on user input data.
- The system uses natural language processing and machine learning techniques to analyze user information and generate personalized health advice.

## Features
- User data collection (personal info, medical history, diet, exercise, sleep patterns, stress levels)
- Natural Language Processing (NLP) for text analysis
- Machine Learning model for generating personalized recommendations
- SQLite database for storing user data and recommendations
- Customized recommendations in categories: stress management, sleep optimization, exercise adjustment, and nutrition optimization

## Requirements
- Python 3.7+
- SQLite3
- Required Python packages:
  - spacy
  - numpy
  - scikit-learn

## Installation

1. Clone the repository:
  - `git clone https://github.com/yourusername/health-recommendations-system.git`
- cd health-recommendations-system

2. Install required packages:
- pip install spacy numpy scikit-learn

3. Download the spaCy English language model:
- python -m spacy download en_core_web_sm

## Usage
Run the main script:
- `python personalized_health_recommendation.py`
- Follow the prompts to enter user information and receive personalized health recommendations.

## System Components:
1. `HealthMetricsExtractor`: Extracts relevant features from user input text.
2. `RecommendationEngine`: Generates personalized health recommendations using a trained machine learning model.
3. `HealthRecommendationSystem`: Manages user interactions, data storage, and recommendation retrieval.

## Database:
The system uses SQLite to store user data and recommendations. Two tables are created:
- `users`: Stores user information
- `recommendations`: Stores generated recommendations for each user

## Customization:
- Update the `get_personalized_recommendations` method in the `RecommendationEngine` class to add new recommendation categories or modify existing ones.

## Note:
- This system is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## License:
- [MIT License](https://opensource.org/licenses/MIT)

## Contributing:
- Contributions, issues, and feature requests are welcome. If you want to contribute, feel free to check the [issues page](https://github.com/yourusername/health-recommendations-system/issues).

## Author
- [Melisa Sever]
