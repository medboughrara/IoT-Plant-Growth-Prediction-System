from src.models.plant_growth_predictor import PlantGrowthPredictor
from src.utils.iot_monitor import IoTPlantMonitor
from src.utils.visualization import (
    plot_correlation_matrix,
    plot_feature_importance,
    plot_confusion_matrix
)

def train_and_evaluate():
    """Train and evaluate the plant growth prediction model"""
    # Initialize the predictor
    predictor = PlantGrowthPredictor()

    # Load and preprocess data
    df = predictor.load_and_preprocess_data('data/Advanced_IoT_Dataset.csv')

    # Perform feature analysis
    feature_importance, correlation_matrix = predictor.feature_analysis(df)
    
    # Plot feature analysis
    plot_correlation_matrix(correlation_matrix)
    plot_feature_importance(feature_importance)

    # Prepare data for machine learning
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = predictor.prepare_data(df)

    # Train models and select the best one
    best_model, cv_scores = predictor.train_models(X_train, y_train)

    # Perform hyperparameter tuning
    best_params = predictor.hyperparameter_tuning(X_train, y_train)

    # Evaluate the model
    accuracy, predictions, cm = predictor.evaluate_model(X_test, y_test)
    plot_confusion_matrix(cm, predictor.label_encoder.classes_)

    # Save the model
    predictor.save_model('models/plant_growth_model.pkl')

    return predictor

if __name__ == "__main__":
    predictor = train_and_evaluate()
    print("Model training completed and saved successfully!")
