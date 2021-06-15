import MBD
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import cv2




def demo_MBD_distance2d_gray_scale_image():
    I = Image.open('/lrde/home2/movn/Documents/Deep/Dahunet/unet_intersection/results/unet_1_fuse/9.png').convert('L')
    I = np.array(I, dtype = np.uint8)

    


    seed_pos = [130, 53]
    S = np.zeros((I.shape[0], I.shape[1]), np.uint8)
    S[seed_pos[0]][seed_pos[1]] = 255

    des_pos = [105, 16]
    des_pos1 = [108, 40]
    des_pos2 = [107,87]
    Des = np.zeros((I.shape[0], I.shape[1]), np.uint8)
    Des[des_pos[0]][des_pos[1]] = 2
    Des[des_pos1[0]][des_pos1[1]] = 3
    Des[des_pos2[0]][des_pos2[1]] = 4




    D1 = MBD.geodesic_shortest_all(I,S,Des)
    cv2.imwrite('abc.png', (D1/2+I/2).astype(np.uint8))
    plt.imshow(D1/2 + I/2)
    plt.show()


if __name__ == '__main__':

    demo_MBD_distance2d_gray_scale_image()

