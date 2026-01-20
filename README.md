# Neural Networks: Comprehensive Educational Repository

## Project Overview

This repository contains a systematic exploration of fundamental neural network concepts, implementation methodologies, and practical applications. The materials comprehensively document the transition from classical machine learning models to modern deep learning architectures, with emphasis on understanding core mathematical foundations and implementation details across multiple frameworks.

## Academic Context

This collection represents educational materials developed as part of formal coursework in machine learning and deep learning. The materials have been curated and enhanced to facilitate comprehensive understanding of neural network theory, architecture, and practical implementation.

## Repository Structure

### Foundational Concepts

**01_Neurons_and_Layers.ipynb** - Introduction to Neural Network Fundamentals
- Exploration of neuron architecture and layer composition
- Comparison of neural network models with classical regression and logistic regression
- Introduction to TensorFlow and Keras frameworks
- Analysis of activation functions and their mathematical properties
- Hands-on implementation of single and multi-layer networks

**02_CoffeeRoasting_TF.ipynb** - Neural Network Implementation with TensorFlow
- Practical application of neural networks to binary classification
- Dataset: coffee roasting quality assessment based on temperature and duration
- Implementation using TensorFlow's Sequential API
- Exploration of model training, evaluation, and prediction
- Visualization of decision boundaries and model performance

**03_CoffeeRoasting_Numpy.ipynb** - NumPy-Based Neural Network Implementation
- Building neural networks from fundamental principles using NumPy
- Forward propagation implementation without high-level frameworks
- Dense layer construction with custom activation functions
- Multi-layer network architecture development
- Comparative analysis between framework-based and custom implementations

### Advanced Topics

**Binary classification on NN.ipynb** - Handwritten Digit Recognition (Binary Classification)
- Application of neural networks to image recognition tasks
- Binary classification of handwritten digits (0 and 1)
- Dataset: Modified MNIST subset (1000 training examples, 20×20 pixel images)
- Multiple implementation approaches: TensorFlow, NumPy, and vectorized NumPy
- Model architecture exploration and optimization strategies
- Broadcasting techniques in numerical computing

**C2_W2_Relu.ipynb** - ReLU Activation Function Analysis
- Deep exploration of Rectified Linear Unit (ReLU) activation function
- Non-linear behavior and computational advantages
- Network design using ReLU for complex decision boundaries
- Interactive visualization and parameter optimization
- Comparison with sigmoid and other activation functions

**C2_W2_SoftMax.ipynb** - Softmax Function and Multiclass Classification
- Softmax function theory and implementation
- Categorical cross-entropy loss analysis
- Multiclass classification implementation
- Comparison between naive and numerically stable implementations
- Application to multi-class classification problems

## Key Learning Outcomes

1. **Theoretical Foundations**: Comprehensive understanding of neural network mathematics, including forward propagation, activation functions, and loss functions

2. **Implementation Mastery**: Practical experience implementing neural networks across multiple frameworks (TensorFlow, NumPy) and understanding trade-offs

3. **Activation Functions**: In-depth analysis of sigmoid, ReLU, and softmax functions, including their mathematical properties and computational implications

4. **Architecture Design**: Understanding of network layer composition, weight initialization, and hyperparameter selection

5. **Classification Tasks**: Hands-on experience with binary and multiclass classification problems using real-world examples

## Technical Stack

- **Python 3.7+**: Primary programming language
- **NumPy**: Numerical computing and array operations
- **TensorFlow 2.0+**: Deep learning framework
- **Keras**: High-level neural network API (integrated with TensorFlow)
- **Matplotlib**: Data visualization and result presentation

## Methodology

The repository follows a pedagogical progression:

1. **Sequential Complexity**: Progresses from simple linear models to complex multi-layer networks
2. **Comparative Implementation**: Demonstrates equivalent concepts across NumPy and TensorFlow
3. **Mathematical Rigor**: All implementations include clear mathematical formulations
4. **Visual Learning**: Extensive use of plots and visualizations to illustrate concepts
5. **Practical Application**: Each concept is immediately applied to realistic datasets

## Datasets and Examples

- **Housing Price Prediction**: Linear regression with single neuron
- **Coffee Roasting Classification**: Binary classification with real-world physical parameters
- **Handwritten Digit Recognition**: Image classification using MNIST subset
- **Synthetic Classification Data**: Multi-class problems with generated datasets

## Activation Functions Covered

- **Linear Activation**: For regression tasks
- **Sigmoid Activation**: For binary classification and understanding non-linearity
- **ReLU (Rectified Linear Unit)**: For deep networks and complex feature learning
- **Softmax Activation**: For multiclass probability distributions

## Loss Functions and Optimization

- **Mean Squared Error (MSE)**: For regression tasks
- **Binary Cross-Entropy**: For binary classification
- **Sparse Categorical Cross-Entropy**: For multiclass classification
- **Adam Optimizer**: For efficient gradient-based optimization

## Pedagogical Features

- Clear markdown documentation explaining concepts before implementation
- Progressive code examples building from simple to complex
- Visualizations showing model behavior and decision boundaries
- Output demonstrations showing expected results
- Mathematical notation with proper LaTeX formatting
- Comments explaining non-obvious implementation details

## Numerical Stability Considerations

The materials explicitly address numerical stability issues:
- Comparison between naive and stable implementations of softmax
- Discussion of log-sum-exp trick for numerical stability
- Proper handling of activation functions in final layers with custom loss functions

## Attribution and Acknowledgments

**Primary Contribution**: Sandesh Bhatta

**Educational Material Credit**: Coursera Deep Learning Specialization

This repository represents an enhanced compilation and documentation of learning materials from Coursera's formal deep learning curriculum. All content has been organized, documented, and refined to provide a comprehensive educational resource for understanding neural networks from foundational principles through advanced implementation techniques.

## References

- LeCun, Y., Cortes, C., & Burges, C. J. (2010). MNIST Handwritten Digit Database
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press
- TensorFlow Documentation: https://www.tensorflow.org/
- Chollet, F. (2015). Keras: Deep Learning Library for Python

## Usage Notes

Each notebook is designed as a standalone educational unit while maintaining thematic connections to other notebooks. The materials are suitable for:

- Self-paced learning of neural network fundamentals
- Supplementary materials for formal deep learning courses
- Reference implementation for neural network algorithms
- Starting point for advanced neural network experimentation

---

*Last Updated: January 2026*
