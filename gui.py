from tensorflow.keras.models import model_from_json
import numpy as np
import cv2


def EmotionDetectionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

EMOTION_LIST=['Angry','Disgusted','Fearful','Happy','Neutral','Sad','Surprised']


model = EmotionDetectionModel("model_a.json","model_weights.h5")

c=cv2.VideoCapture(0)

while True:
    r,frame=c.read()
    
    if not r:
        break
    
    frame=cv2.resize(frame,(1280,720))
    faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    num_faces=faces.detectMultiScale(gray_frame,1.3,5)
    
    for (x,y,w,h) in num_faces:
        cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(0,255,0),4)
        roi=gray_frame[y:y+h,x:x+w]
        roi=cv2.resize(roi,(48,48))
        roi=roi[np.newaxis,:,:,np.newaxis]
        
        pred=EMOTION_LIST[np.argmax(model.predict(roi))]
        
        cv2.putText(frame,pred,(x+5,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        
    cv2.imshow("Emotion Dectection",frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
c.release()
cv2.destroyAllWindows()