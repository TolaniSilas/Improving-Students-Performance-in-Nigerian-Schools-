from catboost import CatBoostClassifier


model_path = "catboost_model.cbm" 
model = CatBoostClassifier()

print(f"\nLoading the saved model from: {model_path}...")
model.load_model(model_path)
print("The saved model has been loaded successfully!\n")
  

# Define the model class.
class Model:
    
    # model_path = "catboost_model.cbm"  # Class-level attribute for model path.
    # model = None  # Class-level attribute for the model instance.
    
    # def load_model(self):
    #     """
    #     Load the model from the specified path if it hasn't been loaded already,
    #     and print a confirmation message.
    #     """
            
    #     if Model.model is None:
            
    #         try:
    #             print(f"\nLoading the saved model from: {self.model_path}...")
    #             Model.model = CatBoostClassifier()
    #             Model.model.load_model(self.model_path)
    #             print("The saved model has been loaded successfully!\n")
                
    #         except Exception as e:
    #             print(f"An error occurred while loading the model: {e}")
                
    #     else:
    #         print("Model is already loaded.\n")
    
    def prediction(self, input_data):
        """
        Use the loaded model to make predictions on the given input data.
        
        Args:
            input_data: The input data for making predictions.
        
        Returns:
            The predicted output based on the input data.
        """
        try:
            
            # Load model if not already loaded.
            # self.load_model() 
            print("\nMaking predictions on the input data...")
            predictions = model.predict(input_data)
            
            return predictions
        
        except Exception as e:
            print(f"An error occurred while making predictions: {e}")
    
    def prediction_probability(self, input_data):
        """
        Use the loaded model to calculate prediction probability on the given input data.
        
        Args:
            input_data: The user's input data.
        
        Returns:
            The probability output based on the input data.
        """
        
        try:
            
            # Load model if not already loaded.
            # self.load_model() 
            print("\nMaking prediction probability on the input data...")
            prediction_proba = model.predict_proba(input_data)
            
            return prediction_proba
        
        except Exception as e:
            print(f"An error occurred while calculating prediction probability: {e}")
