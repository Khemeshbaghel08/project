# Quora queries with QQAD

This project involves developing a state-of-the-art question-answering model using the Quora Question Answer Dataset (QQAD). The goal is to create an AI system capable of understanding and generating accurate responses to user queries.

## Directory Structure



## Setup

1. Clone the repository:

    git clone <repository-url>
   
  

2. Create and activate the virtual environment:
   
    python -m venv venv
    venv\Scripts\activate  
   
    

3. Install the required dependencies:
   
    pip install -r requirements.txt
    

4. Run the data cleaning script:
    
    python src/data_cleaning.py
    

5. Train the model:

    python src/train_model.py
    

## Project Overview

### Data Exploration and Preprocessing

Detailed exploration and preprocessing steps can be found in the notebooks located in the `notebooks/` directory.

### Model Training

We fine-tuned a BERT model on the QQAD. Training scripts and configurations are located in the `src/` directory.

### Evaluation

We evaluated the model using various metrics like ROUGE, BLEU, and F1-score. Evaluation scripts can be found in the `src/` directory.

### Visualization

Visualizations of data distribution, feature importance, and model performance are included in the notebooks.



