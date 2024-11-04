import cv2
# Read the PFM file
for i in range(150):
    image = cv2.imread(f'../python/out/pfms/loop_output{i}.pfm', cv2.IMREAD_UNCHANGED)
    # # Display the image
    #cv2.imshow('PFM Image', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    image *= 255
    cv2.imwrite(f'../python/out/jpgs/output_jpg{i}.jpg', image)

