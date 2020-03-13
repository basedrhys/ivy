import cv2
from joblib import Parallel, delayed
import multiprocessing

from tracker import add_new_blobs, remove_duplicates, update_blob_tracker
from detectors.detector import get_bounding_boxes
from util.detection_roi import get_roi_frame, draw_roi
from util.logger import get_logger
from counter import attempt_count
import datetime
import time

logger = get_logger()
num_cores = multiprocessing.cpu_count()

def update_blob_tracker_queue(in_queue, out_queue):
    '''
    Update a blob's tracker object.
    '''
    while True:
        blob, blob_id, frame = in_queue.get(True)
        box = blob.tracker.update(frame)
        
        blob.num_consecutive_tracking_failures = 0
        blob.update(box)
        logger.debug('Vehicle tracker updated.', extra={
            'meta': {
                'label': 'TRACKER_UPDATE',
                'vehicle_id': blob_id,
                'bounding_box': blob.bounding_box,
                'centroid': blob.centroid,
            },
        })

        out_queue.put((blob_id, blob))

class VehicleCounter():

    def __init__(self, initial_frame, detector, tracker, droi, show_droi, mcdf, mctf, di, counting_lines, draw_counts, use_droi):
        self.frame = initial_frame # current frame of video
        self.detector = detector
        self.tracker = tracker
        self.droi = droi # detection region of interest
        self.show_droi = show_droi
        self.use_droi = use_droi
        self.mcdf = mcdf # maximum consecutive detection failures
        self.mctf = mctf # maximum consecutive tracking failures
        self.di = di # detection interval
        self.counting_lines = counting_lines

        self.blobs = {}
        self.f_height, self.f_width, _ = self.frame.shape
        self.frame_count = 0 # number of frames since last detection
        self.counts = {counting_line['label']: {} for counting_line in counting_lines} # counts of vehicles by type for each counting line
        self.draw_counts = draw_counts

        # create blobs from initial frame
        droi_frame = get_roi_frame(self.frame, self.droi)
        _bounding_boxes, _classes, _confidences = get_bounding_boxes(droi_frame, self.detector)
        self.blobs = add_new_blobs(_bounding_boxes, _classes, _confidences, self.blobs, self.frame, self.tracker, self.mcdf)

        self.tracking_count = 0
        self.tracking_average = 0
        self.detection_count = 0
        self.detection_average = 0

        self.use_own_KCF_impl = False
        if self.use_own_KCF_impl:
            import multiprocessing as mp
            self.in_queue = mp.Queue()
            self.out_queue = mp.Queue()
            self.pool = mp.Pool(4, update_blob_tracker_queue, (self.in_queue, self.out_queue))

    def get_blobs(self):
        return self.blobs

    def count_queue(self, frame):
        self.frame = frame

        for blob_id, blob in self.blobs.items():
            self.in_queue.put((blob, blob_id, self.frame))

        num_blobs = len(self.blobs)
        processed_blobs = 0
        while processed_blobs < num_blobs:
            blob_id, blob = self.out_queue.get(True)
            blob, self.counts = attempt_count(blob, blob_id, self.counting_lines, self.counts)
            self.blobs[blob_id] = blob

            # remove blob if it has reached the limit for tracking failures
            if blob.num_consecutive_tracking_failures >= self.mctf:
                del self.blobs[blob_id]
            
            processed_blobs += 1

        if self.frame_count >= self.di:
            # rerun detection
            if self.use_droi:
                droi_frame = get_roi_frame(self.frame, self.droi)
            else:
                droi_frame = self.frame

            _bounding_boxes, _classes, _confidences = get_bounding_boxes(droi_frame, self.detector)

            self.blobs = add_new_blobs(_bounding_boxes, _classes, _confidences, self.blobs, self.frame, self.tracker, self.mcdf)
            self.blobs = remove_duplicates(self.blobs)
            self.frame_count = 0

        self.frame_count += 1

    def running_average(self, new_time, is_detection=False):
        if is_detection:
            # Unpack the stored average
            total_time = self.detection_average * self.detection_count
            total_time += new_time
            self.detection_count += 1
            self.detection_average = total_time / self.detection_count
        else:
            # Unpack the stored average
            total_time = self.tracking_average * self.tracking_count
            total_time += new_time
            self.tracking_count += 1
            self.tracking_average = total_time / self.tracking_count

    def count(self, frame):
        self.frame = frame

        blobs_list = list(self.blobs.items())
        time0 = time.time()
        # update blob trackers
        blobs_list = [update_blob_tracker(blob, blob_id, self.frame) for blob_id, blob in blobs_list]
        # blobs_list = Parallel(n_jobs=num_cores, prefer='threads')(
        #     delayed(update_blob_tracker)(blob, blob_id, self.frame) for blob_id, blob in blobs_list
        # )
        tracking_time_ms = (time.time() - time0) * 1000
        print("%d - %.3f" % (len(blobs_list), tracking_time_ms))
        self.running_average(tracking_time_ms, is_detection=False)
        self.blobs = dict(blobs_list)

        for blob_id, blob in blobs_list:
            # count vehicle if it has crossed a counting line
            blob, self.counts = attempt_count(blob, blob_id, self.counting_lines, self.counts)
            self.blobs[blob_id] = blob

            # remove blob if it has reached the limit for tracking failures
            if blob.num_consecutive_tracking_failures >= self.mctf:
                del self.blobs[blob_id]


        if self.frame_count >= self.di:
            time0 = time.time()
            # rerun detection
            if self.use_droi:
                droi_frame = get_roi_frame(self.frame, self.droi)
            else:
                droi_frame = self.frame

            _bounding_boxes, _classes, _confidences = get_bounding_boxes(droi_frame, self.detector)
            

            self.blobs = add_new_blobs(_bounding_boxes, _classes, _confidences, self.blobs, self.frame, self.tracker, self.mcdf)
            self.blobs = remove_duplicates(self.blobs)
            self.frame_count = 0
            det_time = float(time.time() - time0) * 1000
            print("\t", det_time)
            self.running_average(det_time, is_detection=True)
            


        self.frame_count += 1

    def visualize(self):
        frame = self.frame
        font = cv2.FONT_HERSHEY_DUPLEX
        line_type = cv2.LINE_AA

        colors = [(255, 255, 255), (255, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255), (0, 255, 0), (0, 255, 255)]
        color_i = 0
        # draw and label blob bounding boxes
        for _id, blob in self.blobs.items():
            this_color = colors[color_i % len(colors)]
            color_i += 1
            (x, y, w, h) = [int(v) for v in blob.bounding_box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), this_color, 2)
            vehicle_label = 'I: ' + _id[:8] \
                            if blob.type is None \
                            else 'I: {0}, T: {1} ({2})'.format(_id[:8], blob.type, str(blob.type_confidence)[:4])
            cv2.putText(frame, vehicle_label, (x, y - 5), font, 0.5, this_color, 1, line_type)

        # draw counting lines
        for counting_line in self.counting_lines:
            cv2.line(frame, counting_line['line'][0], counting_line['line'][1], (255, 0, 0), 3)
            cl_label_origin = (counting_line['line'][0][0], counting_line['line'][0][1] + 35)
            cv2.putText(frame, counting_line['label'], cl_label_origin, font, 1, (255, 0, 0), 2, line_type)

        # show detection roi
        if self.show_droi:
            frame = draw_roi(frame, self.droi)
        
        # For each line, write the cumulative count on the video (note if you're detecting lots
        # of different classes, this may overflow off the bottom of the video).
        if self.draw_counts:
            offset = 1
            for line, objects in self.counts.items():
                cv2.putText(frame, line, (10, 40 * offset), font, 1, (255, 0, 0), 2, line_type)
                offset += 1

                for label, count in objects.items():
                    cv2.putText(frame, "{}: {}".format(label, count), (10, 40 * offset), font, 1, (255, 0, 0), 2, line_type)
                    offset += 1

        return frame
