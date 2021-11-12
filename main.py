import torch 
import cv2
import numpy as np

##################
# https://www.pexels.com/video/video-of-famous-landmark-on-a-city-during-daytime-1721294/ - traffic1.pm4
# https://www.pexels.com/video/time-lapse-footage-of-city-highway-s-vehicular-traffic-at-night-2561431/ - traffic2
# https://www.pexels.com/video/time-lapse-footage-of-clouds-over-an-empty-highway-lined-with-trees-2532566/ - empty-road.mp4
# https://www.pexels.com/video/driving-on-the-city-on-a-rainy-day-4832152/ - rain.mp4
# https://www.pexels.com/video/city-driving-at-night-3555570/ - night.mp4
###################

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/weights/last.pt', force_reload=True)

def predict_video(video: int or str):
    """
        Makes prediction on passed:
            - str - movie (path to movie)
            - int - video from device (number of device - e.g. webcam, usb cam etc.)
    """

    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        result = model(frame)
        
        cv2.imshow('webcam', np.squeeze(result.render()[0]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    predict_video('data/traffic1.mp4')