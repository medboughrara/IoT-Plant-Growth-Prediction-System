from datetime import datetime
import pandas as pd

class IoTPlantMonitor:
    """
    Class for real-world IoT implementation
    Integrates with IoT sensors to make real-time predictions
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def process_sensor_data(self, sensor_data):
        """Process real-time sensor data from IoT devices"""
        try:
            # Validate sensor data
            required_features = self.predictor.feature_columns
            if not all(feature in sensor_data for feature in required_features):
                missing = [f for f in required_features if f not in sensor_data]
                raise ValueError(f"Missing sensor data for: {missing}")

            # Make prediction
            predicted_class, probabilities = self.predictor.predict_single_sample(sensor_data)

            # Create response
            response = {
                'timestamp': datetime.now(),
                'sensor_data': sensor_data,
                'predicted_class': predicted_class,
                'confidence': float(max(probabilities)),
                'all_probabilities': dict(zip(self.predictor.label_encoder.classes_,
                                            probabilities))
            }

            return response

        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now()}

    def batch_process(self, sensor_data_list):
        """Process multiple sensor readings in batch"""
        return [self.process_sensor_data(sensor_data) for sensor_data in sensor_data_list]
