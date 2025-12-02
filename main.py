# main.py
from data_processing import load_and_process_data
from feature_engineering import prepare_features_and_target
from model_training_and_evaluation import train_and_evaluate

def main():
    print("\n🚀 HR Analytics – Employee Attrition Prediction")
    print("==============================================\n")

    # Step 1: Load and Clean Data
    print("📌 Step 1: Loading and cleaning dataset...")
    df = load_and_process_data()
    print(f"✔ Dataset loaded. Shape: {df.shape}\n")

    # Step 2: Feature Engineering
    print("📌 Step 2: Preparing features and target variable...")
    X, y = prepare_features_and_target(df)
    print(f"✔ Feature matrix shape: {X.shape}, Target vector shape: {y.shape}\n")

    # Step 3: Train and Evaluate Models
    print("📌 Step 3: Training models and evaluating performance...")
    best_model, metrics_dict, metrics_df = train_and_evaluate(X, y)

    print("\n✅ All steps completed successfully!\n")

    print("=== Final Metrics Summary ===")
    for model_name, metrics in metrics_dict.items():
        print(f"\n{model_name}:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.3f}")

    print(f"\n🏆 Best model saved as: best_attrition_model.pkl")
    print("\n🎉 Pipeline executed successfully!")

if __name__ == "__main__":
    main()