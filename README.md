# Personality Classification with KNN Algorithm

## Overview
This project implements a K-Nearest Neighbors (KNN) algorithm to classify personality types based on the Myers-Briggs Type Indicator (MBTI) framework. The model analyzes responses to personality questionnaire data and predicts one of the 16 personality types.

## Dataset
The project uses a comprehensive dataset (`16P.csv`) containing:
- **60,000 survey responses** from personality questionnaires
- **60 question features** with responses ranging from -3 to +3 (Likert scale)
- **16 personality type classifications** (ESTJ, ENTJ, ESFJ, etc.)

### Personality Types Mapping
The 16 Myers-Briggs personality types are encoded as follows:
- 0: ESTJ (The Supervisor)
- 1: ENTJ (The Commander)
- 2: ESFJ (The Provider)
- 3: ENFJ (The Giver)
- 4: ISTJ (The Inspector)
- 5: ISFJ (The Nurturer)
- 6: INTJ (The Mastermind)
- 7: INFJ (The Counselor)
- 8: ESTP (The Doer)
- 9: ESFP (The Performer)
- 10: ENTP (The Visionary)
- 11: ENFP (The Champion)
- 12: ISTP (The Craftsman)
- 13: ISFP (The Composer)
- 14: INTP (The Thinker)
- 15: INFP (The Idealist)

## Features
- **K-Nearest Neighbors Implementation**: Custom implementation of KNN algorithm with Euclidean distance calculation
- **5-Fold Cross Validation**: Robust evaluation using cross-validation technique
- **Data Normalization**: Comparison between normalized and non-normalized data performance
- **Multiple K Values**: Testing with different neighbor sizes (1, 3, 5, 7, 9)
- **Performance Metrics**: Comprehensive evaluation using accuracy, precision, and recall
- **Confusion Matrix**: Detailed analysis of classification performance

## Algorithm Implementation

### Core Components
1. **Distance Calculation**: Uses NumPy's `linalg.norm()` for efficient Euclidean distance computation
2. **Neighbor Selection**: Sorts distances and selects k-nearest neighbors
3. **Majority Voting**: Predicts personality type based on majority vote among neighbors
4. **Performance Evaluation**: Calculates TP, FP, FN, TN for confusion matrix analysis

### Key Functions
- `get_neighbors(neighbor_size, test)`: Finds k-nearest neighbors for a test instance
- Data preprocessing with normalization option
- Cross-validation implementation for robust testing

## Performance Analysis

### Key Findings
- **Optimal K Value**: K=7 provides the best balance of accuracy and efficiency
- **Normalization Impact**: Normalized data shows faster processing with comparable accuracy
- **Cross-Validation Results**: 5-fold cross-validation ensures robust performance evaluation
- **Error Analysis**: Wrong predictions primarily occur due to boundary cases in feature space

### Performance Metrics
The model evaluates performance using:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positive rate for each personality type
- **Recall**: Sensitivity for each personality type
- **Confusion Matrix**: Detailed breakdown of classification results

## Files Structure
```
├── knn.ipynb              # Main Jupyter notebook with implementation
├── 16P.csv               # Dataset with personality survey responses
├── 16p-Mapping.txt       # Personality types mapping reference
├── LICENSE               # Project license
└── README.md            # Project documentation
```

## Usage

### Prerequisites
```python
import numpy as np
import pandas as pd
```

### Running the Code
1. Open `knn.ipynb` in Jupyter Notebook or JupyterLab
2. Ensure `16P.csv` is in the same directory
3. Run all cells sequentially to execute the full analysis

### Data Loading
```python
data = pd.read_csv("16P.csv", encoding='cp1252')
del data[data.columns[0]]  # Remove response ID column
```

## Results Interpretation

### Best Performance Configuration
- **K Value**: 7 neighbors
- **Data Processing**: Normalized features
- **Validation**: 5-fold cross-validation
- **Processing Speed**: Significantly faster with normalized data

### Error Analysis
The model shows varying performance across different K values:
- **K=1**: Higher variance, susceptible to outliers
- **K=3,5,7**: Better balance through majority voting
- **K=9**: May include too distant neighbors

## Technical Details

### Distance Calculation
The implementation uses vectorized operations for efficient distance computation:
```python
dist = np.linalg.norm((test - train_data), axis=1)
```

### Normalization
Min-Max normalization is applied:
```python
data_norm = (independent_variable - np.min(independent_variable)) / 
            (np.max(independent_variable) - np.min(independent_variable))
```

## Future Improvements
- **Feature Selection**: Identify most predictive personality questions
- **Hyperparameter Tuning**: Automated K value optimization
- **Alternative Distance Metrics**: Manhattan, Cosine similarity
- **Ensemble Methods**: Combine multiple classifiers
- **Deep Learning**: Neural network approach for comparison

## License
This project is available under the license specified in the LICENSE file.

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for improvements and bug fixes.

## Acknowledgments
- Myers-Briggs Type Indicator framework
- Dataset contributors for personality survey responses
- NumPy and Pandas libraries for efficient data processing
