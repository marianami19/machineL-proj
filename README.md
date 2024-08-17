# machineL-proj


Here's a `README.md` file for your project based on the provided code:

---

# Custom Model Evaluation with MTEB

This project evaluates a sentence-transformer model using the MTEB (Massive Text Embedding Benchmark) framework. It includes steps for model installation, evaluation, and visualization of results.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualization](#visualization)
- [License](#license)

## Installation

To set up the environment and install the necessary packages, run the following command:

```bash
!pip install mteb sentence-transformers
```

Ensure that you have a valid Hugging Face token. You can log in using the following command:

```python
from huggingface_hub import login
login(token='YOUR_HF_TOKEN')
```

Replace `YOUR_HF_TOKEN` with your actual Hugging Face token.

## Usage

1. **Import the necessary libraries:**

    ```python
    import mteb
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import login
    ```

2. **Log in to Hugging Face:**

    ```python
    login(token='YOUR_HF_TOKEN')
    ```

3. **Define the sentence-transformers model:**

    ```python
    model_name = "average_word_embeddings_komninos"  # or another model of your choice
    model = SentenceTransformer(model_name)
    ```

4. **Define and initialize tasks:**

    ```python
    tasks = mteb.get_tasks(tasks=["Banking77Classification"])
    evaluation = mteb.MTEB(tasks=tasks)
    ```

5. **Run the evaluation:**

    ```python
    results = evaluation.run(model, output_folder=f"results/{model_name}")
    ```

6. **Print and load results:**

    ```python
    import json
    results_file_path = './results/average_word_embeddings_komninos/sentence-transformers__average_word_embeddings_komninos/21eec43590414cb8e3a'
    with open(results_file_path, 'r') as file:
        results_data = json.load(file)
    print(json.dumps(results_data, indent=4))
    ```

## Results

The evaluation provides various metrics such as accuracy, F1-score, and weighted F1-score. Here’s how to print overall and individual experiment scores:

```python
print(f"Overall Accuracy: {results_data['scores']['test'][0]['accuracy']:.2f}")
print(f"Overall F1-score: {results_data['scores']['test'][0]['f1']:.2f}")
print(f"Overall Weighted F1-score: {results_data['scores']['test'][0]['f1_weighted']:.2f}")

print("\nScores from Individual Experiments:")
for idx, experiment in enumerate(results_data['scores']['test'][0]['scores_per_experiment']):
    print(f"Experiment {idx+1}:")
    print(f" Accuracy: {experiment['accuracy']:.2f}")
    print(f" F1-score: {experiment['f1']:.2f}")
    print(f" Weighted F1-score: {experiment['f1_weighted']:.2f}")
```

## Visualization

The project includes a function to visualize test scores across experiments:

```python
import matplotlib.pyplot as plt

def visualize_test_scores(test_scores):
    experiments = test_scores[0]['scores_per_experiment']
    accuracy_scores = []
    f1_scores = []
    f1_weighted_scores = []

    for experiment in experiments:
        accuracy_scores.append(experiment['accuracy'])
        f1_scores.append(experiment['f1'])
        f1_weighted_scores.append(experiment['f1_weighted'])

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    axes[0].plot(accuracy_scores, marker='o', linestyle='-', color='b')
    axes[0].set_title('Accuracy per Experiment')
    axes[0].set_xlabel('Experiment')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid(True)

    axes[1].plot(f1_scores, marker='o', linestyle='-', color='g')
    axes[1].set_title('F1 Score per Experiment')
    axes[1].set_xlabel('Experiment')
    axes[1].set_ylabel('F1 Score')
    axes[1].grid(True)

    axes[2].plot(f1_weighted_scores, marker='o', linestyle='-', color='r')
    axes[2].set_title('F1 Weighted Score per Experiment')
    axes[2].set_xlabel('Experiment')
    axes[2].set_ylabel('F1 Weighted Score')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

visualize_test_scores(results_data["scores"]["test"])
```

