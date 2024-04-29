import os
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

COLOR_MAPPING = [(200, 200, 200), (149, 206, 178), (107, 180, 70), (249, 237, 178), (219, 244, 109), (131, 121, 52), (64, 219, 163), (110, 234, 226), (24, 213, 134), (157, 253, 223), 
                 (164, 228, 62), (120, 156, 117), (63, 101, 45), (16, 75, 25), (98, 166, 47), (24, 165, 49), (39, 247, 227), (149, 134, 109), (82, 103, 67), (219, 213, 173), 
                 (132, 172, 165), (16, 109, 217), (245, 120, 134), (72, 26, 89), (233, 202, 86), (190, 112, 165), (201, 39, 11), (93, 220, 39), (84, 107, 224), (156, 160, 37), 
                 (49, 157, 57), (178, 99, 168), (120, 191, 111), (224, 182, 194), (161, 4, 11), (189, 3, 80), (226, 47, 245), (42, 216, 57), (125, 96, 84), (21, 113, 53), 
                 (79, 178, 72), (7, 244, 70), (44, 20, 243), (173, 153, 247), (237, 30, 7), (153, 212, 156), (102, 69, 46), (141, 207, 175), (254, 57, 170), (230, 75, 204), 
                 (212, 105, 165), (197, 41, 65), (86, 23, 121), (174, 206, 147), (101, 146, 121), (173, 2, 17), (236, 173, 157), (240, 123, 11), (211, 163, 141), (51, 38, 74), 
                 (59, 95, 176), (98, 61, 98), (42, 180, 64), (116, 55, 242), (0, 199, 80), (105, 82, 16), (132, 104, 108), (92, 1, 109), (160, 182, 209), (169, 83, 11), 
                 (207, 177, 54), (244, 47, 1), (59, 59, 113), (88, 23, 199), (156, 187, 237), (74, 14, 206), (237, 249, 105), (230, 173, 18), (2, 74, 112), (160, 191, 131), 
                 (239, 151, 73), (143, 173, 55), (108, 155, 185), (103, 206, 199), (240, 240, 205), (180, 199, 78), (196, 49, 115), (188, 166, 103), (57, 239, 226), (95, 44, 92), 
                 (141, 49, 208), (120, 67, 13), (197, 197, 138), (17, 5, 137), (20, 136, 248), (162, 93, 20), (96, 18, 11), (33, 130, 189), (18, 229, 73), (65, 36, 160)]

class Instance():
    """Object instance that is being tracked in the scene"""
    def __init__(self, pred_center, contour, instance_id, pixel_label):
        self.pred_center = pred_center      # predicted center of object (x, y)
        self.contour = contour
        self.instance_id = instance_id      # unique identifier for object
        self.skipped_frames = 0             # number of frames where this object wasn't seen
        self.trace = []                     # path of centroids
        self.pixel_label = pixel_label      # in the instance image that is used as input, we use variable to track what pixel value that was


class InstanceTracker():
    """Tracks multiple instances over instance-segmented images"""
    def __init__(self, dist_thresh=50, max_frames_to_skip=5):
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.min_size = 300
        self.instances = [] # list of instances (Instance object)
        self.instance_id_count = 0

    def find_instance_centers(self, mask):
        # Get unique labels in the mask
        unique_labels = np.unique(mask)
        
        centers = []
        largest_contours = []
        center_to_pixel_label = {}
        
        # Iterate through each unique label
        for label in unique_labels:
            if label == 0:
                continue  # Skip background
            
            # Create a mask for the current label
            label_mask = np.uint8(mask == label)
            
            contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_c = contours[0]
            for c in contours:
                area = cv2.contourArea(c)
                if area > cv2.contourArea(max_c):
                    max_c = c
                    
            if cv2.contourArea(max_c) < self.min_size: continue
            
            # Calculate moments for the labeled region
            moments = cv2.moments(max_c)
            
            # Calculate centroid
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])

            centers.append((int(cX), int(cY)))
            largest_contours.append(label_mask)
            center_to_pixel_label[(int(cX), int(cY))] = label

        return centers, largest_contours, center_to_pixel_label

    def calculate_cost_matrix_iou(self, contours_curr):
        num_prev = len(self.instances)
        num_curr = len(contours_curr)
        
        cost_matrix = np.zeros((num_prev, num_curr))
        
        for i in range(num_prev):
            for j in range(num_curr):
                cost_matrix[i][j] = 1 - (np.logical_and(self.instances[i].contour.astype(bool), contours_curr[j].astype(bool)).sum() / # iou
                                        np.logical_or(self.instances[i].contour.astype(bool), contours_curr[j].astype(bool)).sum())
                
        return np.array(cost_matrix)

    def calculate_cost_matrix_dist(self, centers_curr):
        num_prev = len(self.instances)
        num_curr = len(centers_curr)
        
        cost_matrix = np.zeros((num_prev, num_curr))
        
        for i in range(num_prev):
            for j in range(num_curr):
                cost_matrix[i][j] = np.linalg.norm(np.array(self.instances[i].pred_center) - np.array(centers_curr[j]))
        
        return np.array(cost_matrix)

    def update_instances(self, folder):
        images = load_images_from_folder(folder)
        num_images = len(images)
        
        # Create an empty list to hold instance-labeled images
        instance_labeled_images = []
        
        # Initialize centers with the first image
        centers_prev, contours_prev, center_to_pixel_label = self.find_instance_centers(images[0])
        for i in range(len(centers_prev)):
            instance = Instance(centers_prev[i], contours_prev[i], i, center_to_pixel_label[centers_prev[i]])
            self.instances.append(instance)
            self.instance_id_count += 1
        
        for i in tqdm(range(1, num_images)):
            curr_img = images[i]
            
            # Find centers of instances in the current image
            centers_curr, contours_curr, center_to_pixel_label = self.find_instance_centers(curr_img)
            
            # Calculate cost matrix based on the distance between instance centers
            cost_matrix = self.calculate_cost_matrix_dist(centers_curr) + self.calculate_cost_matrix_iou(contours_curr)
            
            # Apply Hungarian algorithm for instance tracking
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            assignment = []
            for _ in range(len(self.instances)):
                assignment.append(-1)
            row_ind, col_ind = linear_sum_assignment(cost_matrix) # row = detection (curr), col = tracking (prev instance list)
            for i in range(len(row_ind)):
                assignment[row_ind[i]] = col_ind[i]
                self.instances[row_ind[i]].trace.append(self.instances[row_ind[i]].pred_center)
                self.instances[row_ind[i]].pred_center = centers_curr[col_ind[i]]
                self.instances[row_ind[i]].pixel_label = center_to_pixel_label[centers_curr[col_ind[i]]]
                    
            # Add unassigned detections to tracking list
            for i in range(len(centers_curr)):
                if i not in assignment:
                    instance = Instance(centers_curr[i], contours_curr[i], self.instance_id_count, pixel_label=center_to_pixel_label[centers_curr[i]])
                    self.instances.append(instance)
                    self.instance_id_count += 1
            
            # Identify tracks with no assignment   
            unassigned_tracks = []
            dist_thresh = 160
            for i in range(len(assignment)):
                if (assignment[i] != -1):
                    # check for cost distance threshold.
                    # If cost is very high then delete the track
                    if (cost_matrix[i][assignment[i]] > dist_thresh):
                        assignment[i] = -1
                        unassigned_tracks.append(i)
                    pass
                else:
                    self.instances[i].skipped_frames += 1
                    self.instances[i].pixel_label = -1
                    
            # Remove tracks that haven't been detected in a while
            del_tracks = []
            for i in range(len(self.instances)):
                if (self.instances[i].skipped_frames > self.max_frames_to_skip):
                    del_tracks.append(i)
            if len(del_tracks) > 0:  # only when skipped frame exceeds max
                for id in del_tracks:
                    if id < len(self.instances):
                        del self.instances[id]
                        del assignment[id]
                    else:
                        print("ERROR: id is greater than length of tracks")
            
            # Draw instance labels on the current image
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2RGB)
            copy_image = curr_img.copy()
            for j, instance in enumerate(self.instances):
                cX, cY = instance.pred_center
                if j == 0:
                    copy_image[np.all(copy_image == (0, 0, 0), axis=-1)] = COLOR_MAPPING[0]
                elif instance.pixel_label != -1:
                    copy_image[np.all(copy_image == (instance.pixel_label, instance.pixel_label, instance.pixel_label), axis=-1)] = COLOR_MAPPING[instance.instance_id]
                elif instance.pixel_label == -1:
                    copy_image[np.all(copy_image == (instance.pixel_label, instance.pixel_label, instance.pixel_label), axis=-1)] = COLOR_MAPPING[0]
                    
                copy_image = cv2.circle(copy_image, (cX, cY), radius=2, color=(255, 255, 255), thickness=-1)
                cv2.putText(copy_image, str(instance.instance_id), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # contours, _ = cv2.findContours(instance.contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # cv2.drawContours(copy_image, contours, -1, (0, 0, 255), 5)
            
            # Append the instance-labeled image to the list
            instance_labeled_images.append(copy_image)
            
            # # Update centers + contours for the next iteration
            # centers_prev = [centers_curr[idx] for idx in col_ind]
            # contours_prev = [contours_curr[idx] for idx in col_ind]
            
        return instance_labeled_images


# OS helper functions

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def save_images_to_folder(images, output_folder):
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(output_folder, f'instance_labeled_{i+1:03d}.jpg'), img)


if __name__=='__main__':
    # Folder containing images with instance labels
    input_folder = '/home/anjashep-frog-lab/Research/vdbfusion_mapping/vdbfusion/examples/notebooks/semantic-kitti-odometry/dataset/sequences/00/image_2_instances'
    # Output folder for instance-labeled images
    output_folder = '/home/anjashep-frog-lab/Desktop/instances/'
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create tracker
    tracker = InstanceTracker(dist_thresh=100)
    
    # Perform instance tracking
    instance_labeled_images = tracker.update_instances(input_folder)
    
    # Save instance-labeled images to the output folder
    save_images_to_folder(instance_labeled_images, output_folder)
    
    print("Instance labeling complete. Images saved in:", output_folder)