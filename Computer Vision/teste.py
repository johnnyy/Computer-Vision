import argparse
import cv2
import numpy as np
import open3d
import matplotlib.pyplot as plt
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Reconstruct the3D map from the two input stereo images. Output will be saved in output.ply')
    parser.add_argument("--image-left", dest="image_left",required=True, help="Input image captured from the left")
    parser.add_argument("--image-right", dest="image_right",required=True,help="Input image captured from the right")
    parser.add_argument("--output-file", dest="output_file",required=True,help="Output filename (without the extension) wherethe point cloud will be saved")
    return parser

def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1,3), colors])
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')
        
if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    image_left = cv2.imread(args.image_left)
    image_right = cv2.imread(args.image_right)
    output_file = args.output_file + '.ply'
    if image_left.shape[0] != image_right.shape[0] or image_left.shape[1] != image_right.shape[1]:
        
        raise TypeError("Input images must be of the same size")
        
image_left = cv2.pyrDown(image_left)
image_right = cv2.pyrDown(image_right)
win_size = 1
min_disp = 16
max_disp = min_disp * 9
num_disp = max_disp - min_disp
# Needs to be divisible by 16
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,numDisparities = num_disp,uniquenessRatio = 10,speckleWindowSize = 100,speckleRange = 64,disp12MaxDiff = 1,P1 = 8*3*win_size**2,P2 = 32*3*win_size**2)
#stereo.setSpeckleWindowSize(500)
disparity_map = stereo.compute(image_left,image_right).astype(np.float32) / 16.0

#stereo = cv2.StereoBM_create(numDisparities=64, # Máximo de disparidades (em pixels) a serem testada 
                                                # acima de minDisparity. Perde esta qte de pixels no lado esq.
                                                # Múltiplo de 16. Flor da direita só aparece com 128.
  #                           blockSize=5) # Tamanho do bloco a ser comparado (ímpar). Menos ruído->Menos detalhes
#stereo.setMinDisparity(4) # Menor disparidade a ser testada. Default: 0
#stereo.setSpeckleRange(4) # Máxima disparidade permitida em cada componente conexa. Será multiplicado por 16.
                          # Ao avaliar os vizinhos de um pixel, conecta apenas de a disparidade for <= ao range.
#stereo.setSpeckleWindowSize(40) # Maior tam. de região suave que será considerada ruído (e removida).
# Mais detalhes em help(cv2.StereoSGBM_create) e help(cv2.StereoMatcher)

#disparity_map = stereo.compute(image_left,image_right)

h, w = image_left.shape[:2]
focal_length = 0.5*w
# Perspective transformation matrix
Q = np.float32([[1, 0, 0, -w/2.0],
[0,-1, 0, h/2.0],
[0, 0, 0, -focal_length],
[0, 0, 1, 0]])
points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
colors = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
mask_map = disparity_map > disparity_map.min()
output_points = points_3D[mask_map]
output_colors = colors[mask_map]
print ("\nCreating the output file ...\n")
create_output(output_points, output_colors, output_file)
pcd_load = open3d.read_point_cloud("final.ply")
open3d.draw_geometries([pcd_load])


#plt.imshow(disparity_map,'gray')
