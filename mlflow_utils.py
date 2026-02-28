import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def start_mlflow_experiment(experiment_name):
    """Sets the experiment name."""
    mlflow.set_experiment(experiment_name)

def log_model_experiment(model, model_name, params, metrics, confusion_matrix_data, classes):
    """Logs parameters, metrics, model, and artifacts (charts)."""
    with mlflow.start_run(run_name=model_name):
        # Log Parameters
        mlflow.log_params(params)
        
        # Log Metrics
        mlflow.log_metrics(metrics)
        
        # Log Model
        mlflow.sklearn.log_model(model, artifact_path=model_name)
        
        # Create and log Confusion Matrix chart
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_data, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title(f"Confusion Matrix: {model_name}")
        
        # Save plot as artifact
        chart_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(chart_path)
        mlflow.log_artifact(chart_path)
        plt.close(fig)

def register_best_model(model_uri, model_name):
    """Registers the best model in the MLflow Model Registry."""
    mlflow.register_model(model_uri, model_name)