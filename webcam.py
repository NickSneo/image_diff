import cv2 

def capture(file_name):
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:    
            
            check, frame = webcam.read()
            # print(check) #prints true as long as the webcam is running 
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'): 
                cv2.imwrite(filename=file_name, img=frame)
                webcam.release()
                filename = file_name
                img_new = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
                img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                print("Image saved!")
                return filename 

            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                return None

        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            return None         

    