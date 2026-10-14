
# IGZO/MgO Synaptic Transistor — Machine Learning Project
## Neuromorphic Digit Recognition System

### 1. Project Overview

This project develops a machine learning-based neuromorphic digit recognition system using an IGZO/MgO synaptic transistor. It combines experimentally obtained transistor characteristics with machine learning models to predict device behavior and simulate synaptic weight updates for digit recognition.

The system includes transistor parameter extraction, drain current prediction, frequency-dependent EPSC gain modeling, and digit recognition using conventional and device-informed learning approaches.

---

### 2. Project Structure

```text
igzo_project/
├── data/
│   ├── IDVD_IDBG_IGZO_MgO.csv
│   └── filterchar_IGZO_MgO_new.csv
│
├── models/
│   ├── device_params.pkl
│   ├── poly_model.pkl
│   ├── nn_model.pkl
│   ├── gain_model.pkl
│   ├── igzo_digit_model.pkl
│   └── igzo_TRUE_digit_model.pkl
│
├── 1_train_model.py
├── 2_digit_recognition.py
├── 2b_digit_recognition_IGZO_TRUE.py
├── 3_app.py
├── requirements.txt
└── README.md
```

**Directory and file descriptions:**

- `data/`: Contains experimental transistor transfer-characteristic data and frequency-dependent EPSC data.
- `models/`: Stores trained machine learning models and extracted device parameters.
- `1_train_model.py`: Extracts transistor parameters and trains device prediction models.
- `2_digit_recognition.py`: Implements digit recognition using a conventional learning approach.
- `2b_digit_recognition_IGZO_TRUE.py`: Implements digit recognition using transistor-inspired synaptic weight updates.
- `3_app.py`: Launches the graphical application for drawing and recognizing digits.
- `requirements.txt`: Lists the Python dependencies required to run the project.

---

### 3. Machine Learning Models

| Model file | Description |
|---|---|
| `device_params.pkl` | Stores extracted transistor parameters, such as threshold voltage (\(V_{th}\)), ON/OFF current ratio (\(I_{ON}/I_{OFF}\)), and subthreshold swing (SS), where supported by the experimental data. |
| `poly_model.pkl` | Polynomial regression model for predicting drain current (\(I_{DS}\)) from gate voltage (\(V_{GS}\)) and any other configured input features. |
| `nn_model.pkl` | Neural network model for learning nonlinear transistor characteristics. |
| `gain_model.pkl` | Model for predicting excitatory postsynaptic current (EPSC) gain as a function of input frequency. |
| `igzo_digit_model.pkl` | Digit recognition model that incorporates IGZO-related weight quantization. |
| `igzo_TRUE_digit_model.pkl` | Digit recognition model that uses transistor-derived potentiation and depression behavior for synaptic weight updates. |

**Important distinction:** The device prediction models are trained using the project's experimental data. The digit recognition models combine that device information with a machine learning classification task. Their exact degree of hardware fidelity depends on how the device characteristics are implemented in the training code.

---

### 4. Installation and Setup

**Prerequisites**

- Python 3.10 or later
- Internet access for the initial MNIST dataset download
- The experimental CSV files included in the `data/` directory

**Install dependencies**

Open a terminal in the project root directory and execute:

```bash
pip install -r requirements.txt
```

Ensure that Python is correctly installed and accessible from your system's PATH.

---

### 5. Execution Workflow

#### Step 1: Train the Device Models

```bash
python 1_train_model.py
```

This step processes the experimental transistor data, extracts supported device parameters, trains the drain current prediction models, and saves the resulting model files.

#### Step 2: Run IGZO-Informed Digit Recognition

```bash
python 2b_digit_recognition_IGZO_TRUE.py
```

This is the primary experimental workflow. It uses transistor-derived potentiation and depression characteristics to simulate synaptic weight updates during learning.

The objective is to investigate how experimentally derived device behavior can influence the learning process and digit recognition performance.

#### Step 3: Run the Conventional Baseline (Optional)

```bash
python 2_digit_recognition.py
```

This provides a comparison against a conventional digit recognition approach. Its accuracy and learning behavior can be compared with the device-informed implementation under consistent experimental conditions.

#### Step 4: Launch the Graphical Application

```bash
python 3_app.py
```

The application provides a graphical interface where users can draw a digit and obtain the model's predicted class.

---

### 6. Understanding IGZO-Informed Synaptic Learning

**Conventional learning**

A conventional gradient-based optimizer updates model weights using a mathematical optimization rule, such as:

\[
W_{\text{new}}=W_{\text{old}}-\eta\nabla_W L
\]

where \(W\) represents the model weights, \(\eta\) is the learning rate, and \(L\) is the loss function.

**IGZO-informed learning**

The device-informed implementation instead uses transistor-derived potentiation and depression characteristics to simulate synaptic weight changes.

Conceptually:

```python
weight = igzo.potentiate(weight)
weight = igzo.depress(weight)
```

These calls are illustrative representations of the project's device model; their exact behavior depends on the implementation in the Python source files.

Potentiation represents an increase in synaptic conductance, while depression represents a decrease. The measured device characteristics are used to model these changes rather than relying exclusively on idealized mathematical weight updates.

**Conductance states**

The project description identifies 402 conductance states derived from the experimental dataset `IDVD_IDBG_IGZO_MgO.csv`. This count should be interpreted as the number of sampled data points or modeled states only if the preprocessing and device-model implementation support that interpretation.

The purpose is to explore how experimentally observed device characteristics can be incorporated into a computational learning system.

---

### 7. Troubleshooting

**Issue: `ModuleNotFoundError`**

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Issue: `FileNotFoundError` for model files**

Run the training script before executing the digit recognition or GUI scripts:

```bash
python 1_train_model.py
```

Verify that the generated files are saved in the expected `models/` directory.

**Issue: The graphical application does not launch**

Update Pillow:

```bash
pip install --upgrade pillow
```

**Issue: MNIST dataset download is slow or fails**

An internet connection is required for the initial dataset download if MNIST has not already been cached locally. Retry the download and verify network connectivity.

---

### 8. Research Reference

The project is inspired by the following research paper:

*Low-Temperature Solution-Processed In₂O₃ Synaptic Transistors.*

*IEEE Transactions on Electron Devices*, Vol. 73, No. 1, January 2026.

The bibliographic details and the relationship between the referenced paper and the specific IGZO/MgO device should be verified before publication or formal academic submission.

---

### 9. Project Objective

The primary objective is to investigate the integration of experimentally characterized synaptic transistor behavior with machine learning for neuromorphic digit recognition.

By combining transistor characterization, nonlinear device modeling, and synaptic learning simulations, the project explores the potential of device-informed learning approaches for future energy-efficient neuromorphic computing systems.