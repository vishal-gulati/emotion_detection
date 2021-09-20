try:
    import numpy as np
    import cv2
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D
    import os
    import os
    import sys
    import pandas as pd
except Exception as e:
    print(e)

oldname = sys.argv[1]
newname=sys.argv[1]+".jpg"
os.rename(oldname,newname)

model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model1.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))
model1.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(1024, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(7, activation='softmax'))
model1.load_weights('data/model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
frame=cv2.imread(newname)
facecasc = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
img = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0)
prediction = model1.predict(img)
maxindex = int(np.argmax(prediction))
print(emotion_dict[maxindex])
os.remove(newname)