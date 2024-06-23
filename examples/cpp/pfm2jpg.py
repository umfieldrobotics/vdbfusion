import cv2
# Read the PFM file
for i in range(50):
    image = cv2.imread(f'loop_output{i}.pfm', cv2.IMREAD_UNCHANGED)
    # # Display the image
    # cv2.imshow('PFM Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image *= 100
    cv2.imwrite(f'jpg_folder/output_jpg{i}.jpg', image)
