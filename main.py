from ultralytics import YOLO
from datetime import datetime, timedelta
from time import time, sleep
import cv2
#from gpiozero import OutputDevice



class AISectaVision:
    def __init__(self, weights, spray_duration, time_delay, zone_detection):
        self.cam = self.initialize_camera()
        self.model = self.load_model(weights)
        self.spray_duration = spray_duration    # spray duration
        self.time_delay = time_delay            # time delay after spraying to start detection
        self.detected = False                   
        self.time_detected = None
        self.zone_detection = self.calculate_detection_area(zone_detection)
        # self.sprayer = OutputDevice(RELAY_PIN, active_high=True, initial_value=False)


    def load_model(self, path):
        model = YOLO(path)
        return model
    

    def initialize_camera(self):
        cam = cv2.VideoCapture(0)
        return cam

    
    def calculate_detection_area(self, threshold):
        w = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        xmin = int((50 - threshold / 2) / 100 * w)
        xmax = int((50 + threshold / 2) / 100 * w)

        return (xmin, xmax)


    def detect_pests(self, frame):
        results = self.model(frame, imgsz=256, stream=True)
        pest_count = 0
        pest_cx = []

        for result in results:
            pest_count = len(result.boxes)

            for box in result.boxes:
                x1 = box.xyxy[0][0]
                y1 = box.xyxy[0][1]
                x2 = box.xyxy[0][2]
                y2 = box.xyxy[0][3]
                
                cxy = (int((x1+x2) / 2), int((y1+y2)/2))
                pest_cx.append(cxy[0])

                cv2.putText(frame, "+", (cxy[0],cxy[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
            frame = result.plot()
        
        self.check_detection_zone(pest_cx)
        
        return frame


    def check_detection_zone(self, pest_cx):
        xmin, xmax = self.zone_detection

        if len(pest_cx) and not self.detected:
            for x in pest_cx:
                if xmin < x < xmax:
                    self.detected = True
                    self.time_detected = datetime.now() + timedelta(self.spray_duration)
                    #self.sprayer.on()
                    return
            self.detected = False


    def check_spray_duration(self):
        now = datetime.now()
        time_diff = (now - self.time_detected).seconds

        if time_diff > self.spray_duration:
            # self.sprayer.off()
            # 
            if time_diff > self.spray_duration + self.time_delay:
                self.detected = False


    def process_frame(self, frame):
        if not(self.detected):
            frame = self.detect_pests(frame)
        else:
            self.check_spray_duration()
        
        xmin, xmax = self.zone_detection
        h = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        cv2.rectangle(frame, (xmin, 0), (xmax, h), (0,255,0), 1)

        return frame


    def run(self):
        while self.cam.isOpened():
            ret, frame = self.cam.read()

            if not(ret):
                return

            res = self.process_frame(frame)
            res = cv2.flip(res, 1)
            cv2.imshow("AI Secta Vision", res)

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break


        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AISectaVision(
        weights='model14j.onnx',
        spray_duration=3,
        time_delay=2,
        zone_detection=20
    )
    app.run()    
