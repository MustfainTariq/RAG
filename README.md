
```markdown
# Retrieval Augmented Generation (RAG) System

## Overview
This project is a Python-based system designed to process, manage, and retrieve data from a repository of Master's theses. It leverages data manipulation libraries and natural language processing (NLP) techniques to provide sophisticated querying capabilities, enhancing research accessibility and efficiency.

## Installation

### Prerequisites
- Python 3.8+
- Pandas
- Langchain
- Ollama

### Setup
Clone the repository to your local machine:
```bash
git clone (https://github.com/MustfainTariq/RAG-CHATBOT.git)
cd RAG-CHATBOT
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

### Loading Data
To load data from the Excel files, run:
```python
python load_data.py
```

### Data Preprocessing
To preprocess the data:
```python
python preprocess_data.py
```

### Data Aggregation
To aggregate the data into a single DataFrame:
```python
python aggregate_data.py
```

### Text Processing with Langchain
For text processing and generating vector space models:
```python
python text_processing.py
```

### Retrieval and Question Answering with Ollama
To use the system for querying:
```python
python query_system.py
```

## Example Queries
Here are some example queries you can perform with the system:
```python
python query_system.py --query "What are the latest trends in NLP research?"
```

## Acknowledgments
This project was submitted to Dr. Akhtar Jamil in the Department of Computer Science at the National University of Computer and Emerging Sciences, Islamabad. Special thanks to the department.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
```
