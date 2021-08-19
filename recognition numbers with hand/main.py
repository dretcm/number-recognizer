import cv2
from Model import Model
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    	print("Cannot open camera")
    	exit()

model = Model()

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
        _, frame = capture.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        cv2.putText(frame, str(model.predict(img)), (10,80), font, 3, (0,0,0), 10, cv2.LINE_AA)
        
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) == ord('q'):
                break
        
capture.release()
cv2.destroyAllWindows()
