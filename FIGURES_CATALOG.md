# Neural Networks Repository - Figures Catalog

## Complete Image Resource Guide with Annotations

**Document Purpose**: Comprehensive catalog of all images and screenshots in the Neural Networks educational repository, organized sequentially with descriptive captions and technical notes.

**Creation Date**: January 2026

---

## Table of Contents

1. [Figures Organization](#figures-organization)
2. [Figure Catalog](#figure-catalog)
3. [Image Specifications](#image-specifications)
4. [Usage Guide](#usage-guide)

---

## Figures Organization

### Organization Scheme

All images in the repository are cataloged and organized with:
- **Sequential Numbering**: Figure 1 through Figure 20
- **Descriptive Titles**: Clear identification of content
- **Technical Notes**: Details about context and usage
- **File References**: Original filenames for tracking
- **Related Notebooks**: Links to notebooks where applicable

### Categories

The 20 screenshots fall into the following analytical categories:

- **Foundational Architecture**: Figures 1-7
- **Activation Functions**: Figures 8-12
- **Classification Examples**: Figures 13-18
- **Advanced Topics**: Figures 19-20

---

## Figure Catalog

### **Figure 1: Neural Network Architecture Foundation**
- **Original Filename**: Screenshot (142).png
- **Classification**: Foundational Architecture
- **Related Notebook**: 01_Neurons_and_Layers.ipynb
- **Description**: Introductory diagram showing basic neuron structure and layer composition
- **Technical Notes**: 
  - Illustrates single neuron with weights, bias, and activation
  - Shows connection between input features and output
  - Foundation for understanding multi-layer networks
- **Usage Context**: Referenced in neural network fundamentals section
- **Suggested Caption**: "Basic Neuron Architecture: Input, Weights, Bias, and Activation Function"

---

### **Figure 2: Linear Model Representation**
- **Original Filename**: Screenshot (143).png
- **Classification**: Foundational Architecture
- **Related Notebook**: 01_Neurons_and_Layers.ipynb
- **Description**: Visualization of linear regression as a single neuron without activation
- **Technical Notes**:
  - Demonstrates connection between classical linear regression and neural networks
  - Shows mathematical formulation: z = w·x + b
  - Illustrates why neural networks extend linear models
- **Usage Context**: Comparison section between Course 1 models and neural networks
- **Suggested Caption**: "Linear Regression as a Single Neuron (No Activation)"

---

### **Figure 3: Logistic Regression Model**
- **Original Filename**: Screenshot (144).png
- **Classification**: Foundational Architecture
- **Related Notebook**: 01_Neurons_and_Layers.ipynb
- **Description**: Binary classification using sigmoid activation function
- **Technical Notes**:
  - Shows single neuron with sigmoid activation
  - Demonstrates threshold-based classification
  - Bridge between linear models and neural networks
- **Usage Context**: Binary classification fundamentals
- **Suggested Caption**: "Logistic Regression: Single Neuron with Sigmoid Activation"

---

### **Figure 4: Multi-Layer Network Structure**
- **Original Filename**: Screenshot (145).png
- **Classification**: Foundational Architecture
- **Related Notebook**: 01_Neurons_and_Layers.ipynb
- **Description**: Three-layer neural network architecture diagram
- **Technical Notes**:
  - Shows input layer, hidden layers, and output layer
  - Illustrates feature extraction through multiple layers
  - Demonstrates network depth for complex problems
- **Usage Context**: Introduction to deep neural networks
- **Suggested Caption**: "Multi-Layer Neural Network: Input → Hidden Layers → Output"

---

### **Figure 5: Forward Propagation Process**
- **Original Filename**: Screenshot (146).png
- **Classification**: Foundational Architecture
- **Related Notebook**: 02_CoffeeRoasting_TF.ipynb, 03_CoffeeRoasting_Numpy.ipynb
- **Description**: Detailed forward propagation flow through network layers
- **Technical Notes**:
  - Shows computation at each layer: z = Wx + b, a = g(z)
  - Illustrates how data flows through the network
  - Essential for understanding network behavior
- **Usage Context**: Forward propagation algorithm explanation
- **Suggested Caption**: "Forward Propagation: Mathematical Operations at Each Layer"

---

### **Figure 6: Decision Boundary - Coffee Roasting Data**
- **Original Filename**: Screenshot (147).png
- **Classification**: Foundational Architecture
- **Related Notebook**: 02_CoffeeRoasting_TF.ipynb
- **Description**: Visualization of decision boundary created by neural network on coffee roasting dataset
- **Technical Notes**:
  - X-axis: Temperature (°C)
  - Y-axis: Duration (minutes)
  - Color intensity: Network output probability
  - Shows non-linear decision boundary capability
- **Usage Context**: Practical application visualization
- **Suggested Caption**: "Decision Boundary: Neural Network Classification on Coffee Roasting Data"

---

### **Figure 7: Network Prediction Regions**
- **Original Filename**: Screenshot (148).png
- **Classification**: Foundational Architecture
- **Related Notebook**: 02_CoffeeRoasting_TF.ipynb
- **Description**: Output regions showing network predictions with decision threshold
- **Technical Notes**:
  - Binary classification regions (Good/Bad roast)
  - Threshold typically at 0.5 probability
  - Shows both continuous and discretized outputs
- **Usage Context**: Prediction visualization and threshold analysis
- **Suggested Caption**: "Network Output Regions: Classification Predictions with Threshold"

---

### **Figure 8: Sigmoid Activation Function**
- **Original Filename**: Screenshot (152).png
- **Classification**: Activation Functions
- **Related Notebook**: 01_Neurons_and_Layers.ipynb
- **Description**: Graph of sigmoid activation function showing S-shaped curve
- **Technical Notes**:
  - Range: (0, 1) - suitable for probability output
  - Formula: σ(z) = 1 / (1 + e^(-z))
  - Derivative shows vanishing gradient problem in deep networks
  - Historical importance in neural networks
- **Usage Context**: Activation function comparison and explanation
- **Suggested Caption**: "Sigmoid Activation Function: S-Shaped Non-Linear Transformation"

---

### **Figure 9: ReLU Activation Function**
- **Original Filename**: Screenshot (153).png
- **Classification**: Activation Functions
- **Related Notebook**: C2_W2_Relu.ipynb
- **Description**: Graph of Rectified Linear Unit (ReLU) activation function
- **Technical Notes**:
  - Formula: ReLU(z) = max(0, z)
  - Piecewise linear nature
  - Computational efficiency compared to sigmoid
  - Preferred in modern deep networks
- **Usage Context**: Activation function advantages explanation
- **Suggested Caption**: "ReLU Activation Function: Efficient Non-Linearity for Deep Learning"

---

### **Figure 10: ReLU Composition for Complex Shapes**
- **Original Filename**: Screenshot (154).png
- **Classification**: Activation Functions
- **Related Notebook**: C2_W2_Relu.ipynb
- **Description**: Demonstration of how multiple ReLU units create piecewise linear functions
- **Technical Notes**:
  - Shows composability of ReLU functions
  - Multiple linear segments create complex shapes
  - Foundation for multi-layer network expressiveness
  - How deep networks approximate non-linear functions
- **Usage Context**: Understanding network expressiveness
- **Suggested Caption**: "ReLU Composition: Creating Complex Functions from Linear Pieces"

---

### **Figure 11: Softmax Probability Distribution**
- **Original Filename**: Screenshot (155).png
- **Classification**: Activation Functions
- **Related Notebook**: C2_W2_SoftMax.ipynb
- **Description**: Softmax output showing probability distribution over multiple classes
- **Technical Notes**:
  - Formula: softmax(z_j) = e^(z_j) / Σ e^(z_k)
  - Probabilities sum to 1.0
  - Suitable for multiclass classification
  - All-or-nothing effect of softmax
- **Usage Context**: Multiclass classification probability explanation
- **Suggested Caption**: "Softmax Function: Probability Distribution for Multiple Classes"

---

### **Figure 12: Activation Function Comparison**
- **Original Filename**: Screenshot (156).png
- **Classification**: Activation Functions
- **Related Notebook**: C2_W2_Relu.ipynb, 01_Neurons_and_Layers.ipynb
- **Description**: Side-by-side comparison of sigmoid, ReLU, and other activation functions
- **Technical Notes**:
  - Sigmoid: smooth, gradient range [0, 0.25]
  - ReLU: piecewise linear, efficient, zero below 0
  - Comparison helps choose appropriate activation
  - Trade-offs between smoothness and efficiency
- **Usage Context**: Activation function selection guidance
- **Suggested Caption**: "Activation Function Comparison: Properties and Trade-offs"

---

### **Figure 13: MNIST Handwritten Digits Sample**
- **Original Filename**: Screenshot (157).png
- **Classification**: Classification Examples
- **Related Notebook**: Binary classification on NN.ipynb
- **Description**: Random sample of 64 handwritten digits (0s and 1s) from MNIST dataset
- **Technical Notes**:
  - 20×20 pixel grayscale images
  - 1000 training examples in subset
  - 400-dimensional feature vectors (unrolled)
  - Labels shown above each image
- **Usage Context**: Dataset visualization and familiarity
- **Suggested Caption**: "MNIST Digit Dataset: Sample of Handwritten Digits 0 and 1"

---

### **Figure 14: Network Architecture for Digit Recognition**
- **Original Filename**: Screenshot (158).png
- **Classification**: Classification Examples
- **Related Notebook**: Binary classification on NN.ipynb
- **Description**: Three-layer network design for binary digit classification
- **Technical Notes**:
  - Input: 400 units (20×20 pixels)
  - Hidden 1: 25 units with sigmoid
  - Hidden 2: 15 units with sigmoid
  - Output: 1 unit with sigmoid (binary)
  - Total parameters: 10,431
- **Usage Context**: Model architecture specification
- **Suggested Caption**: "Network Architecture for Digit Recognition: 400→25→15→1"

---

### **Figure 15: Network Predictions on Test Data**
- **Original Filename**: Screenshot (159).png
- **Classification**: Classification Examples
- **Related Notebook**: Binary classification on NN.ipynb
- **Description**: Visualization of network predictions on test set
- **Technical Notes**:
  - Shows image and predicted probability
  - Confidence levels for each digit
  - Performance evaluation
  - Error analysis capability
- **Usage Context**: Model evaluation and performance
- **Suggested Caption**: "Network Predictions: Confidence Levels for Digit Classification"

---

### **Figure 16: Confusion Matrix - Binary Classification**
- **Original Filename**: Screenshot (160).png
- **Classification**: Classification Examples
- **Related Notebook**: Binary classification on NN.ipynb
- **Description**: Confusion matrix showing true/false positives and negatives
- **Technical Notes**:
  - TP: Correctly classified 1s
  - TN: Correctly classified 0s
  - FP: False positive rate
  - FN: False negative rate
  - Basis for accuracy, precision, recall metrics
- **Usage Context**: Classification performance evaluation
- **Suggested Caption**: "Confusion Matrix: Classification Performance Analysis"

---

### **Figure 17: Training Progress - Loss Curve**
- **Original Filename**: Screenshot (161).png
- **Classification**: Classification Examples
- **Related Notebook**: 02_CoffeeRoasting_TF.ipynb, Binary classification on NN.ipynb
- **Description**: Loss function value across training epochs
- **Technical Notes**:
  - X-axis: Training epoch number
  - Y-axis: Loss value (cross-entropy)
  - Decreasing trend indicates learning
  - Convergence pattern shows training quality
  - Plateauing suggests model saturation
- **Usage Context**: Training dynamics and convergence
- **Suggested Caption**: "Training Progress: Loss Function Convergence Over Epochs"

---

### **Figure 18: Accuracy Metric Evolution**
- **Original Filename**: Screenshot (162).png
- **Classification**: Classification Examples
- **Related Notebook**: Binary classification on NN.ipynb
- **Description**: Classification accuracy improvement during training
- **Technical Notes**:
  - X-axis: Training epoch
  - Y-axis: Accuracy percentage
  - Increasing curve shows learning effectiveness
  - Gap between training and validation accuracy (overfitting indicator)
  - Final accuracy achieved
- **Usage Context**: Model performance tracking
- **Suggested Caption**: "Accuracy Evolution: Training and Validation Performance"

---

### **Figure 19: NumPy Broadcasting Mechanism**
- **Original Filename**: Screenshot (163).png
- **Classification**: Advanced Topics
- **Related Notebook**: Binary classification on NN.ipynb
- **Description**: Visualization of NumPy broadcasting rules and dimension matching
- **Technical Notes**:
  - Shapes can differ if aligned dimension is 1
  - Automatic expansion along broadcast dimension
  - No memory overhead - virtual expansion
  - Critical for vectorized operations
  - Enables efficient matrix operations
- **Usage Context**: NumPy optimization techniques
- **Suggested Caption**: "NumPy Broadcasting: Efficient Array Operations Without Copying"

---

### **Figure 20: Vectorized Operations Performance**
- **Original Filename**: Screenshot (164).png
- **Classification**: Advanced Topics
- **Related Notebook**: Binary classification on NN.ipynb
- **Description**: Comparison of loop-based vs. vectorized implementation performance
- **Technical Notes**:
  - Loop approach: clear but slow
  - Vectorized approach: optimized, uses BLAS
  - Speedup factor: 100x-1000x typical
  - Essential for large-scale models
  - Demonstrates importance of optimization
- **Usage Context**: Computational efficiency importance
- **Suggested Caption**: "Vectorized vs. Loop Implementation: Performance Comparison"

---

## Image Specifications

### File Format Information

| Property | Specification |
|----------|---------------|
| File Format | PNG (Portable Network Graphics) |
| Color Space | RGB or Grayscale |
| Compression | Lossless |
| Typical Resolution | 800×600 to 1200×800 pixels |
| File Size Range | 50KB - 500KB per image |
| Total Collection Size | ~5-10 MB |

### Naming Convention

**Current Format**: `Screenshot (###).png`
- ### = Sequential number (142-164, with gaps)

**Recommended Standardized Format**: `Figure_##_[Descriptive-Title].png`
- Example: `Figure_01_Neural_Network_Architecture.png`
- Example: `Figure_08_Sigmoid_Activation_Function.png`

### Recommended Renaming Scheme

```
Figure_01_Neural_Network_Architecture.png
Figure_02_Linear_Model_Representation.png
Figure_03_Logistic_Regression_Model.png
Figure_04_Multi_Layer_Network_Structure.png
Figure_05_Forward_Propagation_Process.png
Figure_06_Decision_Boundary_Coffee_Roasting.png
Figure_07_Network_Prediction_Regions.png
Figure_08_Sigmoid_Activation_Function.png
Figure_09_ReLU_Activation_Function.png
Figure_10_ReLU_Composition_Complex_Shapes.png
Figure_11_Softmax_Probability_Distribution.png
Figure_12_Activation_Function_Comparison.png
Figure_13_MNIST_Handwritten_Digits_Sample.png
Figure_14_Network_Architecture_Digit_Recognition.png
Figure_15_Network_Predictions_Test_Data.png
Figure_16_Confusion_Matrix_Binary_Classification.png
Figure_17_Training_Progress_Loss_Curve.png
Figure_18_Accuracy_Metric_Evolution.png
Figure_19_NumPy_Broadcasting_Mechanism.png
Figure_20_Vectorized_Operations_Performance.png
```

---

## Usage Guide

### For Documentation

When referencing images in markdown or papers:

```markdown
[Figure 1: Neural Network Architecture Foundation]
As shown in Figure 1, the basic neuron consists of inputs, weights, 
bias, and an activation function...

[Figure 8: Sigmoid Activation Function]
The sigmoid function (Figure 8) provides smooth probability outputs 
suitable for binary classification...
```

### For Presentations

Suggested organization for slides:
- **Section 1 (Foundations)**: Figures 1-7
- **Section 2 (Activation)**: Figures 8-12
- **Section 3 (Applications)**: Figures 13-18
- **Section 4 (Optimization)**: Figures 19-20

### For Educational Materials

Recommended sequence for student progression:
1. Start with Figures 1-4 (basic concepts)
2. Study Figures 5-7 (mechanisms)
3. Learn Figures 8-12 (activation functions)
4. Apply with Figures 13-18 (real problems)
5. Optimize with Figures 19-20 (efficiency)

### Storage Organization

**Recommended folder structure**:
```
Neural Networks/
├── README.md
├── PROJECT_INDEX.md
├── FIGURES_CATALOG.md (this document)
├── images/
│   ├── Figure_01_Neural_Network_Architecture.png
│   ├── Figure_02_Linear_Model_Representation.png
│   ├── ... (remaining figures)
│   └── Figure_20_Vectorized_Operations_Performance.png
├── notebooks/
│   ├── 01_Neurons_and_Layers.ipynb
│   ├── 02_CoffeeRoasting_TF.ipynb
│   └── ... (remaining notebooks)
```

---

## Integration with Notebooks

### Figure-to-Notebook Mapping

| Notebooks | Primary Figures |
|-----------|-----------------|
| 01_Neurons_and_Layers.ipynb | 1, 2, 3, 4, 5, 8 |
| 02_CoffeeRoasting_TF.ipynb | 6, 7, 17, 18 |
| 03_CoffeeRoasting_Numpy.ipynb | 5, 6, 7 |
| Binary classification on NN.ipynb | 13, 14, 15, 16, 17, 18, 19, 20 |
| C2_W2_Relu.ipynb | 9, 10, 12 |
| C2_W2_SoftMax.ipynb | 11, 12 |

### Markdown Integration Example

```markdown
![Figure 1: Neural Network Architecture]
(./images/Figure_01_Neural_Network_Architecture.png)

*Figure 1: Basic Neuron Architecture showing input features, 
weights, bias term, and activation function producing output.*
```

---

## Quality Standards for Figures

### For Publication Use

- ✓ High resolution (minimum 300 DPI for print)
- ✓ Clear labeling and legends
- ✓ Professional color schemes
- ✓ Consistent style across collection
- ✓ Descriptive captions included

### Best Practices

1. **Clarity**: Each figure should be immediately understandable
2. **Context**: Always provide surrounding explanation
3. **Size**: Optimize for web (72-96 DPI) and print (300 DPI)
4. **Accessibility**: Use alt-text and captions for accessibility
5. **Consistency**: Maintain uniform styling across figures

---

## PDF Export Instructions

### For Converting to PDF

**Using Markdown to PDF Tools**:
1. Use Pandoc: `pandoc FIGURES_CATALOG.md -o FIGURES_CATALOG.pdf`
2. Use VS Code extension: Markdown PDF
3. Use online converter: markdown-to-pdf.com

**Generated PDF Will Include**:
- Table of contents
- All figure descriptions and notes
- Technical specifications
- Usage guidelines
- Integration mappings
- Quality standards
- Professional formatting

**Recommended PDF Settings**:
- Page Size: Letter or A4
- Margins: 1 inch (2.54 cm)
- Font: 11pt for body, 16pt for titles
- Color: Full color recommended for figure visibility

---

## Attribution

**Figures Source**: Neural Networks Educational Repository
**Organization**: This catalog version
**Date**: January 2026
**Educational Context**: Coursera Deep Learning Specialization materials
**Contributor**: Sandesh Bhatta

---

## Version Control

**Document Version**: 1.0
**Last Updated**: January 20, 2026
**Status**: Complete and ready for publication

---

**Notes**: This catalog serves as a comprehensive reference for all visual assets in the repository. It provides structured organization, sequential numbering, detailed annotations, and professional documentation suitable for academic publication or formal educational distribution.

