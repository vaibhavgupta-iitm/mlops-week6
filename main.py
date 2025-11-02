"""
Main pipeline script for IRIS classification with MLflow integration.
This script orchestrates the entire ML pipeline from data loading to model evaluation.
"""

import os
import argparse
import logging
from pathlib import Path

from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from src.dvc_operations import DVCOperations

from mlflow.models import infer_signature

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description='IRIS Classification Pipeline with MLflow')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='iris-dvc-pipeline/v1_data.csv',
                       help='Path to the data file')
    parser.add_argument('--metrics-path', type=str, default='iris-dvc-pipeline/metrics.txt',
                       help='Path to save the metrics')
    parser.add_argument('--augment-data', action='store_true',
                       help='Whether to augment the data')
    
    # DVC arguments
    parser.add_argument('--setup-dvc', action='store_true',
                       help='Setup DVC remote and pull data from GCS')
    parser.add_argument('--version', type=str, default='v1.0',
                       help='DVC version to checkout')
    
    # MLflow arguments
    parser.add_argument('--mlflow-tracking-uri', type=str, default=None,
                       help='MLflow tracking server URI (default: local)')
    parser.add_argument('--mlflow-experiment-name', type=str, default='iris-classification',
                       help='MLflow experiment name')
    
    # Training arguments
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                       help='Enable hyperparameter tuning with GridSearchCV')
    parser.add_argument('--max-depth', type=int, default=3,
                       help='Maximum depth for decision tree (if not tuning)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds for tuning')
    
    # Model registry arguments
    parser.add_argument('--use-mlflow-model', action='store_true',
                       help='Load model from MLflow registry instead of training')
    parser.add_argument('--model-stage', type=str, default='Production',
                       choices=['Production', 'Staging', 'None'],
                       help='Model stage to load from MLflow registry')
    parser.add_argument('--model-version', type=int, default=None,
                       help='Specific model version to load from MLflow registry')
    parser.add_argument('--promote-to-production', action='store_true',
                       help='Promote the trained model to Production stage')
    
    # Comparison mode
    parser.add_argument('--run-comparison', action='store_true',
                       help='Run multiple experiments with different hyperparameters for comparison')
    
    args = parser.parse_args()
    
    try:
        # Initialize components
        data_processor = DataProcessor()
        model_trainer = ModelTrainer(
            max_depth=args.max_depth,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment_name
        )
        dvc_ops = DVCOperations()
        
        # Setup DVC and pull from GCS if requested
        if args.setup_dvc:
            logger.info("Setting up DVC remote and pulling data from GCS...")
            remote_url = "gs://mlops-course-verdant-victory-473118-k0-unique-week2-2/iris-pipeline"
            if not dvc_ops.setup_remote(remote_url):
                logger.error("Failed to setup DVC remote")
                return 1
            
            if not dvc_ops.pull_data(args.data_path):
                logger.error("Failed to pull data from DVC")
                return 1
        
        # Checkout specific version if requested
        if args.version != 'v1.0':
            logger.info(f"Checking out version {args.version}...")
            if not dvc_ops.checkout_version(args.version):
                logger.error(f"Failed to checkout version {args.version}")
                return 1
        
        # Load and validate data
        logger.info("Loading and validating data...")
        data = data_processor.load_data(args.data_path)
        
        if not data_processor.validate_data(data):
            logger.error("Data validation failed")
            return 1
        
        # Augment data if requested
        if args.augment_data:
            logger.info("Augmenting data...")
            data = data_processor.augment_data(data)
        
        # Split data
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = data_processor.split_data(data)
        
        # Handle comparison mode - run multiple experiments
        if args.run_comparison:
            logger.info("Running comparison mode with multiple hyperparameter configurations...")
            run_comparison_experiments(model_trainer, X_train, y_train, X_test, y_test)
            logger.info("Comparison experiments completed! Check MLflow UI for visualization.")
            return 0
        
        # Load model from MLflow registry or train new model
        if args.use_mlflow_model:
            logger.info("Loading model from MLflow registry...")
            try:
                model = model_trainer.load_model_from_mlflow(
                    model_name="iris-classifier",
                    stage=args.model_stage,
                    version=args.model_version
                )
                logger.info("Model loaded successfully from MLflow")
                
                # Evaluate the loaded model
                logger.info("Evaluating loaded model...")
                metrics = model_trainer.evaluate_model(X_test, y_test, log_to_mlflow=False)
                
            except Exception as e:
                logger.error(f"Failed to load model from MLflow: {e}")
                logger.info("Falling back to training a new model...")
                args.use_mlflow_model = False
        
        # Train new model if not loaded from registry
        if not args.use_mlflow_model:
            if args.hyperparameter_tuning:
                logger.info("Training model with hyperparameter tuning...")
                model = model_trainer.train_with_hyperparameter_tuning(
                    X_train, y_train, cv=args.cv_folds
                )
            else:
                logger.info("Training model with single hyperparameter set...")
                model = model_trainer.train_model(X_train, y_train)
            
            # Evaluate model
            logger.info("Evaluating model...")
            metrics = model_trainer.evaluate_model(X_test, y_test, log_to_mlflow=True)
            
            # Promote to production if requested
            if args.promote_to_production:
                logger.info("Promoting model to Production stage...")
                best_run_id = model_trainer.get_best_model_from_experiment(
                    experiment_name=args.mlflow_experiment_name,
                    metric="test_accuracy"
                )
                model_trainer.promote_model_to_production(
                    model_name="iris-classifier",
                    run_id=best_run_id
                )
        
        # Save metrics to file
        logger.info("Saving metrics...")
        model_trainer.save_metrics(metrics, args.metrics_path)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Model F1 score: {metrics['f1_score']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_comparison_experiments(model_trainer, X_train, y_train, X_test, y_test):
    """
    Run multiple experiments with different configurations for comparison.
    
    Args:
        model_trainer: ModelTrainer instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    """
    import mlflow
    
    # Experiment 1: Shallow tree
    logger.info("Experiment 1: Shallow tree (max_depth=2)")
    with mlflow.start_run(run_name="shallow_tree_exp"):
        model_trainer.max_depth = 2
        model_trainer.train_model(X_train, y_train, log_to_mlflow=False)
        
        mlflow.log_param("max_depth", 2)
        mlflow.log_param("experiment_type", "shallow_tree")
        
        metrics = model_trainer.evaluate_model(X_test, y_test, log_to_mlflow=True)
        
        
        input_example = X_train.iloc[:5] if isinstance(X_train, pd.DataFrame) else None
        signature = infer_signature(X_train, y_train)

        mlflow.sklearn.log_model(
            sk_model=model_trainer.model,
            name="model",
            registered_model_name="iris-classifier",
            input_example=input_example,
            signature=signature
        )
    
    # Experiment 2: Medium tree
    logger.info("Experiment 2: Medium tree (max_depth=4)")
    with mlflow.start_run(run_name="medium_tree_exp"):
        model_trainer.max_depth = 4
        model_trainer.train_model(X_train, y_train, log_to_mlflow=False)
        
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("experiment_type", "medium_tree")
        
        metrics = model_trainer.evaluate_model(X_test, y_test, log_to_mlflow=True)

        input_example = X_train.iloc[:5] if isinstance(X_train, pd.DataFrame) else None
        signature = infer_signature(X_train, y_train)
        
        mlflow.sklearn.log_model(
            sk_model=model_trainer.model,
            name="model",
            registered_model_name="iris-classifier",
            input_example=input_example,
            signature=signature
        )
    
    # Experiment 3: Deep tree
    logger.info("Experiment 3: Deep tree (max_depth=8)")
    with mlflow.start_run(run_name="deep_tree_exp"):
        model_trainer.max_depth = 8
        model_trainer.train_model(X_train, y_train, log_to_mlflow=False)
        
        mlflow.log_param("max_depth", 8)
        mlflow.log_param("experiment_type", "deep_tree")
        
        metrics = model_trainer.evaluate_model(X_test, y_test, log_to_mlflow=True)

        input_example = X_train.iloc[:5] if isinstance(X_train, pd.DataFrame) else None
        signature = infer_signature(X_train, y_train)
        
        mlflow.sklearn.log_model(
            sk_model=model_trainer.model,
            name="model",
            registered_model_name="iris-classifier",
            input_example=input_example,
            signature=signature
        )
    
    # Experiment 4: Hyperparameter tuning
    logger.info("Experiment 4: Hyperparameter tuning with GridSearchCV")
    param_grid = {
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }
    model_trainer.train_with_hyperparameter_tuning(X_train, y_train, param_grid=param_grid)
    metrics = model_trainer.evaluate_model(X_test, y_test, log_to_mlflow=True)


if __name__ == "__main__":
    exit(main())