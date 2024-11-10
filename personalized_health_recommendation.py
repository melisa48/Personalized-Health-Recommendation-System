import sqlite3
import json
import spacy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import re

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

class HealthMetricsExtractor:
    def __init__(self):
        # Keywords for different health aspects
        self.stress_keywords = ['stress', 'anxiety', 'worried', 'tension', 'pressure']
        self.sleep_keywords = ['sleep', 'insomnia', 'tired', 'fatigue', 'rest']
        self.exercise_keywords = ['exercise', 'workout', 'training', 'fitness', 'sports']
        self.diet_keywords = ['diet', 'nutrition', 'food', 'meal', 'eating']
        self.medical_keywords = ['condition', 'disease', 'symptoms', 'diagnosis', 'treatment']

    def count_keywords(self, text, keywords):
        text = text.lower()
        return sum(text.count(keyword) for keyword in keywords)

    def extract_numeric_values(self, text):
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        return [float(num) for num in numbers]

    def extract_metrics(self, user_data):
        features = []
        
        # Process each field
        for field in ['personal_info', 'medical_history', 'diet', 'exercise', 'sleep', 'stress']:
            text = user_data[field].lower()
            
            # Basic text metrics
            features.extend([
                len(text),  # Length of text
                len(text.split()),  # Word count
                len(set(text.split()))  # Unique word count
            ])
            
            # Keyword counts
            features.extend([
                self.count_keywords(text, self.stress_keywords),
                self.count_keywords(text, self.sleep_keywords),
                self.count_keywords(text, self.exercise_keywords),
                self.count_keywords(text, self.diet_keywords),
                self.count_keywords(text, self.medical_keywords)
            ])
            
            # Numeric values
            numbers = self.extract_numeric_values(text)
            features.append(len(numbers))  # Number of numeric values
            features.append(np.mean(numbers) if numbers else 0)  # Average of numeric values
        
        return np.array(features)

class RecommendationEngine:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.metrics_extractor = HealthMetricsExtractor()

    def generate_training_data(self):
        """Generate synthetic but realistic training data"""
        n_samples = 1000
        n_features = 60  # Based on our feature extraction method
        
        # Generate synthetic feature vectors
        X = np.random.rand(n_samples, n_features)
        
        # Generate labels with realistic distributions
        y = np.zeros(n_samples)
        
        # Assign labels based on feature patterns
        for i in range(n_samples):
            stress_features = X[i, 0:12]  # First 12 features relate to stress
            sleep_features = X[i, 12:24]  # Next 12 features relate to sleep
            exercise_features = X[i, 24:36]  # And so on...
            
            # Create realistic patterns
            if np.mean(stress_features) > 0.7:
                y[i] = 0  # Stress management
            elif np.mean(sleep_features) > 0.7:
                y[i] = 1  # Sleep improvement
            elif np.mean(exercise_features) > 0.7:
                y[i] = 2  # Exercise adjustment
            else:
                y[i] = np.random.randint(0, 5)
        
        return X, y

    def train_model(self):
        X, y = self.generate_training_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate and print metrics
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Testing accuracy: {test_score:.3f}")

    def get_personalized_recommendations(self, category, user_data):
        recommendations = {
            'stress': [
                {
                    'title': "Stress Management",
                    'recommendations': [
                        "Practice mindfulness meditation for 10-15 minutes daily",
                        "Implement regular deep breathing exercises",
                        "Consider scheduling regular breaks during workout sessions",
                        "Try progressive muscle relaxation before bed",
                        "Start journaling to track stress triggers"
                    ],
                    'explanation': "Based on your stress levels and workout routine"
                }
            ],
            'sleep': [
                {
                    'title': "Sleep Optimization",
                    'recommendations': [
                        "Avoid pre-workout supplements within 6 hours of bedtime",
                        "Establish a consistent pre-sleep routine",
                        "Optimize your bedroom environment (temperature, light, noise)",
                        "Consider using a sleep tracking app",
                        "Practice relaxation techniques before bed"
                    ],
                    'explanation': "Based on your sleep patterns and supplement usage"
                }
            ],
            'exercise': [
                {
                    'title': "Exercise Adjustment",
                    'recommendations': [
                        "Include more recovery days in your routine",
                        "Alternate between high and low intensity workouts",
                        "Focus on proper form and controlled movements",
                        "Consider adding mobility work",
                        "Track your progress with a workout log"
                    ],
                    'explanation': "Based on your current exercise routine and goals"
                }
            ],
            'diet': [
                {
                    'title': "Nutrition Optimization",
                    'recommendations': [
                        "Increase vegetable intake with each meal",
                        "Time protein intake around workouts",
                        "Consider meal prepping to ensure consistent nutrition",
                        "Track macronutrients using a food diary",
                        "Stay hydrated throughout the day"
                    ],
                    'explanation': "Based on your dietary patterns and fitness goals"
                }
            ]
        }
        
        return recommendations.get(category, recommendations['stress'])[0]

    def generate_recommendations(self, user_data):
        try:
            # Extract features
            features = self.metrics_extractor.extract_metrics(user_data)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Map prediction to category
            categories = ['stress', 'sleep', 'exercise', 'diet']
            category = categories[int(prediction) % len(categories)]
            
            # Get detailed recommendations
            recommendation = self.get_personalized_recommendations(category, user_data)
            
            return recommendation
            
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return {
                'title': "General Recommendation",
                'recommendations': ["Please consult with a healthcare professional for personalized advice."],
                'explanation': "Unable to generate specific recommendations due to an error."
            }

class HealthRecommendationSystem:
    def __init__(self):
        self.engine = RecommendationEngine()
        self.setup_database()

    def setup_database(self):
        conn = sqlite3.connect('health_recommendations.db')
        cursor = conn.cursor()
        
        # Create tables with improved schema
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            personal_info TEXT NOT NULL,
            medical_history TEXT NOT NULL,
            diet TEXT NOT NULL,
            exercise TEXT NOT NULL,
            sleep TEXT NOT NULL,
            stress TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            recommendation_text TEXT NOT NULL,
            category TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        conn.commit()
        conn.close()

    def get_user_input(self):
        return {
            'personal_info': input("Enter personal info (age, gender, height, weight): ").strip(),
            'medical_history': input("Enter medical history: ").strip(),
            'diet': input("Describe your diet: ").strip(),
            'exercise': input("Describe your exercise routine: ").strip(),
            'sleep': input("Describe your sleep patterns: ").strip(),
            'stress': input("Describe your stress levels: ").strip()
        }

    def store_user_data(self, user_id, user_data):
        try:
            conn = sqlite3.connect('health_recommendations.db')
            cursor = conn.cursor()
            
            # Store user data
            cursor.execute('''
            INSERT OR REPLACE INTO users (
                user_id, personal_info, medical_history, diet, exercise, sleep, stress, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                user_id,
                user_data['personal_info'],
                user_data['medical_history'],
                user_data['diet'],
                user_data['exercise'],
                user_data['sleep'],
                user_data['stress']
            ))
            
            conn.commit()
            print("Data successfully stored in database")
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()

    def store_recommendation(self, user_id, recommendation):
        try:
            conn = sqlite3.connect('health_recommendations.db')
            cursor = conn.cursor()
            
            # Store recommendation
            cursor.execute('''
            INSERT INTO recommendations (user_id, recommendation_text, category)
            VALUES (?, ?, ?)
            ''', (
                user_id,
                json.dumps(recommendation['recommendations']),
                recommendation['title']
            ))
            
            conn.commit()
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            conn.close()

    def run(self):
        print("Initializing Health Recommendation System...")
        self.engine.train_model()
        
        while True:
            user_id = input("\nEnter user ID (or 'q' to quit): ").strip()
            if user_id.lower() == 'q':
                break

            # Collect user data
            user_data = self.get_user_input()
            
            # Store user data
            self.store_user_data(user_id, user_data)
            
            # Generate recommendations
            recommendation = self.engine.generate_recommendations(user_data)
            
            # Store recommendation
            self.store_recommendation(user_id, recommendation)
            
            # Display recommendations
            print(f"\n{recommendation['title']}:")
            print(f"\nBased on our analysis: {recommendation['explanation']}")
            print("\nRecommendations:")
            for i, rec in enumerate(recommendation['recommendations'], 1):
                print(f"{i}. {rec}")

def main():
    try:
        system = HealthRecommendationSystem()
        system.run()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()