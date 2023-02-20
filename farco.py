import cv2
from utils import show_data, run_haar_cascade_classifier, nothing, predict

def farco():

    camera = cv2.VideoCapture(0) # Nos quedamos con la cámara 0 que es la webcam

    cv2.namedWindow("Output")
    
    cv2.createTrackbar('Positive class threshold','Output', 50, 90, nothing)
    
    while True:
        ret, frame = camera.read()
        
        if not ret:
            raise Exception("Can't get frame. Exiting")

        threshold = cv2.getTrackbarPos('Positive class threshold', 'Output')

        prediction, probability  = predict(frame, threshold)  # Sería bueno una función que retorne bool
                                        # y la probabilidad de la clase

        frame = run_haar_cascade_classifier(frame) 

        frame = show_data(frame, prediction, probability)

        cv2.imshow("Output", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("ESC Pressed. Closing window.")
            break

    camera.release()

    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    farco()