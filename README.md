# Real-Time Arabic Sign Language Recognition

## Project Description
This project implements a deep learning system for recognizing Arabic sign language phrases in real-time using hand landmark detection and LSTM neural networks.

## Dataset
The dataset consists of 7 sign phrases recorded by 5 participants under varying lighting and background conditions.

Each sample is converted into hand landmarks using MediaPipe and standardized to sequences of 30 frames with 63 features per frame.

## Model
The base model uses:
- LSTM layers
- Dense layers
- Softmax classification

The system supports experimentation with different architectures such as:
- Larger LSTM units
- GRU layers
- Dropout regularization
- Different optimizers

## How to Run

1. Create virtual environment:
python3 -m venv venv
source venv/bin/activate

2. Install requirements:
pip install -r requirements.txt

3. Train model:
python train_model.py

## Future Work
- Increase dataset size
- Improve model robustness
- Deploy as a mobile application using TensorFlow Lite