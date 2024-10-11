import argparse
import cv2

from models_training import MatcherFactory  # Importing the MatcherFactory to create matchers based on methods


# Function to draw matches between two images
def draw_matches(image1, kp1, des1, image2, kp2, des2, good_matches=None, bad_matches=None, title=None):
    # If good matches are not provided, compute them using BFMatcher
    if good_matches is None:
        bf = cv2.BFMatcher()  # Create a BFMatcher object
        matches = bf.knnMatch(des1, des2, k=2)  # Find the k-nearest matches

        good_matches, bad_matches = [], []  # Initialize lists for good and bad matches
        for m, n in matches:
            # Classify matches as good or bad based on their distances
            (good_matches if m.distance < 0.6 * n.distance else bad_matches).append([m])

    # Draw the good matches in green and bad matches in red
    img_correct_matches = cv2.drawMatchesKnn(image1, kp1, image2, kp2, good_matches, None, flags=2,
                                             matchColor=(0, 255, 0))
    img_incorrect_matches = cv2.drawMatchesKnn(image1, kp1, image2, kp2, bad_matches[:100], None, flags=2,
                                               matchColor=(255, 0, 0))

    # Combine the images of good and bad matches
    combined_img = cv2.addWeighted(img_correct_matches, 0.5, img_incorrect_matches, 0.5, 0)

    # Show the combined image
    cv2.imshow(title or 'Matching', combined_img)  # Use provided title or default to 'Matching'
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()  # Close the display window


# Main function to execute the feature matching process
def main(method, img1_path, img2_path):
    # Read the images in grayscale
    image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Create the appropriate matcher based on the selected method
    matcher = MatcherFactory.create_matcher(method)

    # If the method is SIFT or ORB, detect and compute keypoints and descriptors
    if method in ['SIFT', 'ORB']:
        kp1, des1 = matcher.detect_and_compute(image1)
        kp2, des2 = matcher.detect_and_compute(image2)
        draw_matches(image1, kp1, des1, image2, kp2, des2)  # Draw matches
    # If the method is SuperGlue, detect and match keypoints
    elif method == 'SuperGlue':
        kp1, kp2, matches = matcher.detect_and_match(image1, image2)

        # Convert keypoints to OpenCV format
        kp1_cv = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in kp1]
        kp2_cv = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in kp2]

        # Prepare good and bad matches based on the matching results
        good_matches = [[cv2.DMatch(i, m, 0)] for i, m in enumerate(matches) if m > -1]
        bad_matches = [[cv2.DMatch(i, m, 0)] for i, m in enumerate(matches) if m <= -1]

        draw_matches(image1, kp1_cv, None, image2, kp2_cv, None, good_matches, bad_matches)  # Draw matches


# Entry point of the script
if __name__ == '__main__':
    # Create an argument parser for command-line inputs
    parser = argparse.ArgumentParser(description='Feature Matching Inference.')
    parser.add_argument('method', choices=['SIFT', 'ORB', 'SuperGlue'],
                        help="Feature detection method")  # Method selection
    parser.add_argument('image1', help="Path to the first image")  # Path to the first image
    parser.add_argument('image2', help="Path to the second image")  # Path to the second image
    args = parser.parse_args()  # Parse the arguments from the command line

    # Call the main function with the provided arguments
    main(args.method, args.image1, args.image2)
