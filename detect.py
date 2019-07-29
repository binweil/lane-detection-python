import numpy as np
import cv2
from keras.models import load_model


class Lanes_Detection():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []
        self.radius_of_curvature = 0
        self.offset = 0

    def compute_offset_from_center(self, line_lt, line_rt, frame_width):
        """
        Compute offset from center of the inferred lane.
        The offset from the lane center can be computed under the hypothesis that the camera is fixed
        and mounted in the midpoint of the car roof. In this case, we can approximate the car's deviation
        from the lane center as the distance between the center of the image and the midpoint at the bottom
        of the image of the two lane-lines detected.

        :param line_lt: detected left lane-line
        :param line_rt: detected right lane-line
        :param frame_width: width of the undistorted frame
        :return: inferred offset
        """
        if line_lt.detected and line_rt.detected:
            line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
            line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
            lane_width = line_rt_bottom - line_lt_bottom
            midpoint = frame_width / 2
            offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
            offset_meter = xm_per_pix * offset_pix
        else:
            offset_meter = -1

        return offset_meter

    def perspective_transform(self,img):
        h, w = img.shape[:2]
        src = np.float32([[w, h-10],    # br
                          [0, h-10],    # bl
                          [500*160/1280, 460*80/720],   # tl
                          [800*160/1280, 460*80/720]])  # tr
        dst = np.float32([[w, h],       # br
                          [0, h],       # bl
                          [0, 0],       # tl
                          [w, 0]])      # tr
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

        left_line = []
        right_line = []
        for y in range(0,len(warped),1): # 80
            minx = 160
            maxx = 0
            for x in range(0,len(warped[1]),1): # 160
                if warped[y][x][1] >= 200:
                    if x < minx:
                        minx = x
                    if x > maxx:
                        maxx = x
            left_line.append([minx, y])
            right_line.append([maxx, y])
        left_line, self.radius_of_curvature = self.polynomial_fit(left_line)
        right_line, _ = self.polynomial_fit(right_line)
        # left_line.sort(axis=0)
        return warped, left_line, right_line

    def polynomial_fit(self, line):
        line = np.array(line)
        coefficients = np.polyfit(line[:,0], line[:,1], 2)
        polynomial = np.poly1d(coefficients)
        x = line[:, 0]
        y = polynomial(line[:, 0])
        radius_of_curvature = ((1 + (2 * coefficients[0] * 0 + coefficients[1]) ** 2) ** 1.5) \
                                / np.absolute(2 * coefficients[0])
        #print("Curvature: ",self.radius_of_curvature)
        return np.array([x*1280/160, y*720/80],np.int), radius_of_curvature

    def draw_line(self, image, line):
        # Re-size to match the original image
        # self.offset = abs((left_line[0,0] + right_line[0,0])/2-640)
        lane_image = cv2.resize(image, (1280,720),interpolation=cv2.INTER_AREA)
        line = line.transpose()
        line = line.reshape((-1, 1, 2))
        cv2.polylines(lane_image, [line], False,(0,255,0),thickness=3,lineType=8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(lane_image, "Curvature:" + str(int(self.radius_of_curvature)) + " meters", (10, 50), font, 2, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(lane_image, "Offset:" + str(int(self.offset)) + " pixel", (10, 100), font, 2, (0, 255, 0), 2,
                    cv2.LINE_AA)
        return lane_image

    def road_lines(self, image):
        """ Takes in a road image, re-sizes for the model,
        predicts the lane to be drawn from the model in G color,
        recreates an RGB image of a lane and merges with the
        original road image.
        """

        lines = []
        left_line = []
        right_line = []
        first_loop = True

        # Get image ready for feeding into model
        small_img = cv2.resize(image, (160,80),interpolation = cv2.INTER_AREA)
        small_img = np.array(small_img)
        small_img = small_img[None,:,:,:]
        # Make prediction with neural network (un-normalize value by multiplying by 255)
        prediction = model.predict(small_img)[0] * 255

        # Add lane prediction to list for averaging
        self.recent_fit.append(prediction)

        # Only using last five for average
        if len(self.recent_fit) > 5:
            self.recent_fit = self.recent_fit[1:]

        # Calculate average detection
        self.avg_fit = np.mean(np.array([i for i in self.recent_fit]), axis = 0)

        # Generate fake R & B color dimensions, stack with G
        blanks = np.zeros_like(self.avg_fit).astype(np.uint8)

        lane_drawn = np.dstack((blanks, self.avg_fit, blanks))

        # Extract Lane positions
        for x in range(0,len(lane_drawn[1]),1): # 160
            miny = 80
            for y in range(0,len(lane_drawn),1): # 80
                if lane_drawn[y][x][1] >= 200:
                    if y < miny:
                        miny = y
            if miny != 80:
                lines.append([x,miny])
                if first_loop:
                    first_loop = False
                elif (first_loop == False) and miny > lines[-2][1]:
                    # right line
                    right_line.append([x,miny])
                elif (first_loop == False) and miny < lines[-2][1]:
                    # left line
                    left_line.append([x,miny])

        # Find Polynomial line fit and draw the polynomial
        left_line, _ = self.polynomial_fit(left_line)
        right_line, _ = self.polynomial_fit(right_line)
        lane_image = self.draw_line(image,left_line)
        lane_image = self.draw_line(lane_image,right_line)
        lane_image = cv2.cvtColor(lane_image,cv2.COLOR_BGR2RGB)

        # Use Perspective Transformation to get curvature
        bird, left_line_bird, right_line_bird = self.perspective_transform(lane_drawn)
        bird = cv2.resize(bird, (1280,720),interpolation = cv2.INTER_AREA)

        lane_drawn = np.dstack((blanks, blanks, blanks))
        bird_line = self.draw_line(lane_image, left_line_bird)
        bird_line = self.draw_line(bird_line, right_line_bird)
        #lane_image = np.hstack([lane_image,bird_line])
        return lane_image


if __name__ == "__main__":
    # Load Keras model
    model = load_model('full_CNN_model_HY.h5')

    lanes = Lanes_Detection()
    cap = cv2.VideoCapture("project_video.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # SCNN accepts BGR format
        frame_ml = lanes.road_lines(frame)
        cv2.imshow("Camera", frame_ml)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
