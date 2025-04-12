import streamlit as st
import os
import numpy as np
import yaml
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
import time
import sys
import pandas as pd

# Add the project root to system path to import custom modules
sys.path.append(os.path.abspath(""))

# Import project modules
from src.textSummarizer.utils.common import read_yaml, create_directories
from src.textSummarizer.logging import logger

# Page configuration
st.set_page_config(
    page_title="Text Summarizer App",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stProgress .st-ee {
        background-color: #4CAF50;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .status {
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .success {
        background-color: #DFF2BF;
        color: #4F8A10;
    }
    .warning {
        background-color: #FEEFB3;
        color: #9F6000;
    }
    .error {
        background-color: #FFBABA;
        color: #D8000C;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Operations", "Model Training", "Evaluation", "Inference"])

# Constants
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# Helper functions
def load_config():
    """Load configuration from YAML files"""
    try:
        config = read_yaml(CONFIG_FILE_PATH)
        params = read_yaml(PARAMS_FILE_PATH)
        return config, params
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None, None

def check_model_exists():
    """Check if trained model exists"""
    try:
        config, _ = load_config()
        model_path = Path(config.model_evaluation.model_path)
        tokenizer_path = Path(config.model_evaluation.tokenizer_path)
        return model_path.exists() and tokenizer_path.exists()
    except:
        return False

def get_model_and_tokenizer():
    """Load model and tokenizer"""
    try:
        config, _ = load_config()
        model_path = config.model_evaluation.model_path
        tokenizer_path = config.model_evaluation.tokenizer_path
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model and tokenizer: {e}")
        return None, None, None

def summarize_text(input_text, model, tokenizer, device):
    """Generate summary using the loaded model"""
    try:
        # Create a summarization pipeline
        summarizer = pipeline(
            "summarization", 
            model=model, 
            tokenizer=tokenizer, 
            device=0 if device == "cuda" else -1
        )
        
        # Generate summary
        summary = summarizer(
            input_text, 
            max_length=150, 
            min_length=30, 
            length_penalty=0.8,
            num_beams=8,
            do_sample=False
        )
        
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def update_yaml_file(file_path, updates):
    """Update YAML file with new values"""
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Update values based on the provided dictionary
        if isinstance(updates, dict):
            for section, section_updates in updates.items():
                if section not in config:
                    config[section] = {}
                for key, value in section_updates.items():
                    config[section][key] = value
        
        with open(file_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        return True
    except Exception as e:
        st.error(f"Error updating YAML file: {e}")
        return False

def run_data_ingestion():
    """Run data ingestion process"""
    from src.textSummarizer.components.data_ingestion import DataIngestion
    from src.textSummarizer.config.configuration import ConfigurationManager
    
    try:
        st.info("Running data ingestion...")
        progress_bar = st.progress(0)
        
        # Create ConfigurationManager and get data ingestion config
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        
        # Create DataIngestion object and run
        progress_bar.progress(25)
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        progress_bar.progress(50)
        data_ingestion.downlaod_file()
        
        progress_bar.progress(75)
        data_ingestion.extract_zip_file()
        
        progress_bar.progress(100)
        st.success("Data ingestion completed successfully!")
        
        # Show data path
        st.info(f"Data downloaded to: {data_ingestion_config.unzip_dir}")
        return True
    except Exception as e:
        st.error(f"Error during data ingestion: {e}")
        return False

def run_data_transformation():
    """Run data transformation process"""
    from src.textSummarizer.components.data_transformation import DataTransformation
    from src.textSummarizer.config.configuration import ConfigurationManager
    
    try:
        st.info("Running data transformation...")
        progress_bar = st.progress(0)
        
        # Create ConfigurationManager and get data transformation config
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        
        # Create DataTransformation object
        progress_bar.progress(25)
        data_transformation = DataTransformation(config=data_transformation_config)
        
        # Run transformation
        progress_bar.progress(50)
        data_transformation.convert()
        
        progress_bar.progress(100)
        st.success("Data transformation completed successfully!")
        
        # Show data path
        st.info(f"Transformed data saved to: {os.path.join(data_transformation_config.root_dir, 'samsum_dataset')}")
        return True
    except Exception as e:
        st.error(f"Error during data transformation: {e}")
        return False

def run_model_trainer():
    """Run model training process"""
    from src.textSummarizer.components.model_trainer import ModelTrainer
    from src.textSummarizer.config.configuration import ConfigurationManager
    
    try:
        st.info("Running model training...")
        progress_bar = st.progress(0)
        
        # Create ConfigurationManager and get model trainer config
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        
        # Create ModelTrainer object
        progress_bar.progress(10)
        model_trainer = ModelTrainer(config=model_trainer_config)
        
        # Run training (this will take time)
        progress_bar.progress(20)
        with st.spinner("Training model... This may take a while"):
            model_trainer.train()
        
        progress_bar.progress(100)
        st.success("Model training completed successfully!")
        
        # Show model path
        st.info(f"Model saved to: {os.path.join(model_trainer_config.root_dir, 'pegasus-samsum-model')}")
        return True
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return False

def run_model_evaluation():
    """Run model evaluation process"""
    from src.textSummarizer.components.model_evaluation import ModelEvaluation
    from src.textSummarizer.config.configuration import ConfigurationManager
    
    try:
        st.info("Running model evaluation...")
        progress_bar = st.progress(0)
        
        # Create ConfigurationManager and get model evaluation config
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        
        # Create ModelEvaluation object
        progress_bar.progress(25)
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        
        # Run evaluation
        progress_bar.progress(50)
        with st.spinner("Evaluating model..."):
            model_evaluation.evaluate()
        
        progress_bar.progress(100)
        st.success("Model evaluation completed successfully!")
        
        # Show evaluation results
        try:
            metrics_df = pd.read_csv(model_evaluation_config.metric_file_name)
            st.subheader("Evaluation Metrics")
            st.dataframe(metrics_df)
        except:
            st.info("No metrics file found.")
        
        return True
    except Exception as e:
        st.error(f"Error during model evaluation: {e}")
        return False

# Pages
def home_page():
    st.title("Text Summarizer Application")
    
    st.markdown("""
    <div class="highlight">
    <h3>Welcome to the Text Summarizer App!</h3>
    <p>This application leverages the power of Pegasus transformer model to generate concise summaries from text.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Overview
    st.header("Project Overview")
    st.write("""
    This application provides a user-friendly interface to interact with a text summarization pipeline:
    
    1. **Data Operations**: Ingest and transform the SAMSUM dataset
    2. **Model Training**: Train the Pegasus model on dialogue summarization
    3. **Evaluation**: Assess the model performance using ROUGE metrics
    4. **Inference**: Generate summaries for your own dialogues
    """)
    
    # System Status
    st.header("System Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Environment")
        st.write(f"Python Version: {sys.version.split()[0]}")
        st.write(f"Torch Version: {torch.__version__}")
        st.write(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name(0)}")
    
    with col2:
        st.subheader("Model Status")
        model_exists = check_model_exists()
        if model_exists:
            st.markdown('<p class="status success">Model is available for inference</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status warning">Model not found. Please train the model first.</p>', unsafe_allow_html=True)

def data_operations_page():
    st.title("Data Operations")
    
    # Data Ingestion
    st.header("1. Data Ingestion")
    st.write("""
    This step downloads the SAMSUM dataset and prepares it for processing.
    The dataset contains dialogues paired with their human-written summaries.
    """)
    
    if st.button("Run Data Ingestion"):
        run_data_ingestion()
    
    # Data Transformation
    st.header("2. Data Transformation")
    st.write("""
    This step tokenizes the dataset and prepares it for model training.
    It converts raw text into tokenized inputs suitable for the Pegasus model.
    """)
    
    # Advanced configuration options
    with st.expander("Advanced Configuration"):
        tokenizer_name = st.text_input("Tokenizer Name", "google/pegasus-cnn_dailymail")
        
        if st.button("Update Configuration"):
            updates = {
                "data_transformation": {
                    "tokenizer_name": tokenizer_name
                }
            }
            if update_yaml_file(CONFIG_FILE_PATH, updates):
                st.success("Configuration updated successfully!")
    
    if st.button("Run Data Transformation"):
        run_data_transformation()

def model_training_page():
    st.title("Model Training")
    
    st.write("""
    This page allows you to train the Pegasus model on the SAMSUM dataset for dialogue summarization.
    You can adjust various training parameters before starting the training process.
    """)
    
    # Training parameters
    st.header("Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.number_input("Number of Training Epochs", min_value=1, max_value=10, value=1)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=8, value=1)
        warmup_steps = st.number_input("Warmup Steps", min_value=100, max_value=1000, value=500)
    
    with col2:
        eval_strategy = st.selectbox("Evaluation Strategy", ["steps", "epoch", "no"])
        eval_steps = st.number_input("Evaluation Steps", min_value=100, max_value=1000, value=500)
        weight_decay = st.slider("Weight Decay", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
    
    # Update configuration
    if st.button("Update Training Configuration"):
        updates = {
            "TrainingArguments": {
                "num_train_epochs": int(epochs),
                "per_device_train_batch_size": int(batch_size),
                "warmup_steps": int(warmup_steps),
                "eval_strategy": eval_strategy,
                "eval_steps": int(eval_steps),
                "weight_decay": float(weight_decay)
            }
        }
        
        if update_yaml_file(PARAMS_FILE_PATH, updates):
            st.success("Training configuration updated successfully!")
    
    # Start training
    st.header("Start Training")
    st.warning("Warning: Training may take a long time depending on your hardware.")
    
    if st.button("Train Model"):
        run_model_trainer()

def evaluation_page():
    st.title("Model Evaluation")
    
    st.write("""
    This page runs evaluation on the trained model using the test set from the SAMSUM dataset.
    The evaluation calculates ROUGE metrics to measure the quality of generated summaries.
    """)
    
    # Check if model exists
    model_exists = check_model_exists()
    
    if not model_exists:
        st.warning("Model not found. Please train the model first.")
    else:
        # Evaluation parameters
        st.header("Evaluation Parameters")
        test_samples = st.slider("Number of Test Samples", min_value=5, max_value=100, value=10)
        
        # Update configuration if needed
        if st.button("Update Evaluation Configuration"):
            # No changes needed to YAML for this parameter, but you could add more later
            st.success("Configuration updated successfully!")
        
        # Run evaluation
        if st.button("Evaluate Model"):
            run_model_evaluation()
        
        # Show previous evaluation results if available
        try:
            config, _ = load_config()
            metrics_file = Path(config.model_evaluation.metric_file_name)
            if metrics_file.exists():
                st.header("Previous Evaluation Results")
                metrics_df = pd.read_csv(metrics_file)
                st.dataframe(metrics_df)
        except:
            pass

def inference_page():
    st.title("Text Summarization")
    
    st.write("""
    Try out the text summarizer by entering a dialogue.
    The model will generate a concise summary of the conversation.
    """)
    
    # Check if model exists
    model_exists = check_model_exists()
    
    if not model_exists:
        st.warning("Model not found. Please train the model first or use the demo.")
        if st.button("Use Demo Mode"):
            st.session_state['demo_mode'] = True
        else:
            return
    
    # Input area
    st.header("Input Dialogue")
    sample_text = """
    Amanda: I need a new phone, my old one is too slow.
    Bob: What about the new iPhone?
    Amanda: Too expensive. I'm thinking about Android.
    Bob: Samsung has some good options.
    Amanda: I heard their cameras are good.
    Bob: Yes, especially the S21. Great camera, good battery, and cheaper than iPhone.
    Amanda: Sounds good. I'll check it out.
    """
    
    user_input = st.text_area("Enter dialogue text", sample_text, height=200)
    
    # Generate summary
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            if model_exists:
                # Load model and tokenizer
                model, tokenizer, device = get_model_and_tokenizer()
                if model and tokenizer:
                    # Generate summary
                    summary = summarize_text(user_input, model, tokenizer, device)
                    
                    # Display summary
                    if summary:
                        st.header("Generated Summary")
                        st.write(summary)
            else:
                # Demo mode with predefined summaries
                time.sleep(2)  # Simulate processing time
                st.header("Generated Summary (Demo)")
                st.write("Amanda needs a new phone as her old one is slow. She's considering Android as iPhone is too expensive. Bob suggests Samsung, mentioning the S21 has a great camera, good battery, and is cheaper than iPhone. Amanda will check it out.")

# Main function to render the appropriate page
def main():
    if page == "Home":
        home_page()
    elif page == "Data Operations":
        data_operations_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Evaluation":
        evaluation_page()
    elif page == "Inference":
        inference_page()

# Run the app
if __name__ == "__main__":
    main()