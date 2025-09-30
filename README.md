# Real-Time-Facial-Emotion-Recognition


This project detects human emotions (Angry, Disgusted, Fearful, Happy, Sad, Neutral, Surprised) from live webcam feed using a CNN built in TensorFlow/Keras.

---

## Features

- Train a CNN on the FER2013 dataset.  
- Detect emotions from live webcam feed.  
- Save and load trained model weights.  

---

## File Structure
Face-Emotion-Detection/  
│  
├── dataset.py # Converts FER2013 CSV to train/test images  
├── train_model.py # Train CNN model   
├── main.py # Run webcam emotion detection  
├── README.md  
└── haarcascade_frontalface_default.xml  

## Dataset

This project uses the FER2013 dataset.

**Download link:**  
[FER2013 CSV on Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013)

**Steps to prepare the dataset:**

1. Download `fer2013.csv` and place it in the project folder.  
2. Run the dataset extraction script:

```bash
python dataset.py
```

## Train the Model

After extracting images, train the CNN model by running `train_model.py`.  

This will create `model.h5` after training.

## Run Webcam Emotion Detection

After training (or if you already have `model.h5`), run `main.py`.  

The webcam feed will open.  
Detected faces will show a rectangle with predicted emotion.  
Press `q` to quit.


