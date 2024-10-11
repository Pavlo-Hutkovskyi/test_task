import cv2
import torch
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor


# Class to perform SIFT (Scale-Invariant Feature Transform) matching
class SIFTMatcher:
    def __init__(self):
        # Create a SIFT detector
        self.sift = cv2.SIFT_create()

    def detect_and_compute(self, image):
        # Detect keypoints and compute descriptors from the input image
        kp, des = self.sift.detectAndCompute(image, None)
        return kp, des


# Class to perform ORB (Oriented FAST and Rotated BRIEF) matching
class ORBMatcher:
    def __init__(self):
        # Create an ORB detector
        self.orb = cv2.ORB_create()

    def detect_and_compute(self, image):
        # Detect keypoints and compute descriptors from the input image
        kp, des = self.orb.detectAndCompute(image, None)
        return kp, des


# Class to perform matching using the SuperGlue algorithm
class SuperGlueMatcher:
    def __init__(self):
        # Configuration for SuperPoint and SuperGlue models
        config = {
            'superpoint': {
                'nms_radius': 4,  # Non-maximum suppression radius
                'keypoint_threshold': 0.0005,  # Threshold for keypoint detection
                'max_keypoints': -1  # Maximum number of keypoints (-1 means unlimited)
            },
            'superglue': {
                'weights': 'outdoor',  # Use pre-trained weights for outdoor scenes
                'sinkhorn_iterations': 20,  # Number of Sinkhorn iterations for matching
                'match_threshold': 0.2,  # Threshold for accepting matches
            }
        }
        # Initialize the SuperGlue matching model
        self.matching = Matching(config).eval()

    def detect_and_match(self, image1, image2):
        # Convert the input images to tensors for the model
        img1_tensor = frame2tensor(image1, torch.device('cpu'))
        img2_tensor = frame2tensor(image2, torch.device('cpu'))

        # Perform matching between the two images using the SuperGlue model
        pred = self.matching({'image0': img1_tensor, 'image1': img2_tensor})

        # Extract keypoints and matches from the prediction
        kp1 = pred['keypoints0'][0].cpu().numpy()  # Keypoints from the first image
        kp2 = pred['keypoints1'][0].cpu().numpy()  # Keypoints from the second image
        matches = pred['matches0'][0].cpu().numpy()  # Matches for the first image

        return kp1, kp2, matches


# Factory class to create matchers based on the specified method
class MatcherFactory:
    @staticmethod
    def create_matcher(method):
        # Create and return the appropriate matcher based on the method name
        if method == 'SIFT':
            return SIFTMatcher()
        elif method == 'ORB':
            return ORBMatcher()
        elif method == 'SuperGlue':
            return SuperGlueMatcher()
        else:
            # Raise an error if the method is unsupported
            raise ValueError("Unsupported method. Please choose from 'SIFT', 'ORB', 'SuperGlue'.")
