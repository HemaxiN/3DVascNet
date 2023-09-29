#3D Retinal Vasculature Quantification
import pandas as pd
from skimage.morphology import binary_erosion, binary_dilation, convex_hull_image, skeletonize_3d, label
from skimage import draw
from skimage.measure import regionprops
import cv2
from skan import Skeleton, summarize
import numpy as np
import os
from scipy import signal
import tifffile
import scipy.ndimage as ndi
from utils.ConcaveHull import concaveHull
from PIL import Image, ImageDraw

masks_dir = r'C:\Users\hemax\Desktop\testing\masks_proc'
resolution_file = r'C:\Users\hemax\Desktop\testing\resolution.xlsx'

resolution = pd.read_excel(resolution_file)

#Function for the computation of the number of bifurcations and end points
def branching_points(branch_data):
    branch_points=0
    end_points=0
    node_origin = branch_data['node-id-src'].values.tolist()
    node_destination = branch_data['node-id-dst'].values.tolist()
    all_nodes = node_origin + node_destination
    aux_list = np.array(all_nodes)
    bpoints_id = []
    epoints_id = []
    for element in np.unique(aux_list):
        occur = all_nodes.count(element)
        if occur>=3:
            branch_points=branch_points+1
            bpoints_id.append(element)
        elif occur==1:
            end_points=end_points+1
            epoints_id.append(element)
    return branch_points, bpoints_id, end_points, epoints_id

def compute_chull(mask):
    mask = np.max(mask, axis=-1)
    x,y = np.shape(mask)
    mask = cv2.resize(mask, (100,100)) #resize the mask to decrease computational cost
    mask[mask!=0] = 255.0
    mask_edges = mask - (ndi.morphology.binary_erosion(mask)*255.0) #obtain the edges to compute concave hull

    #get the coordinates (x,y) of the points belonging to the edges of the mask
    rows, cols = np.where(mask_edges == 255)
    cols = np.expand_dims(cols, axis=-1)
    rows = np.expand_dims(rows, axis=-1)
    points_2d = np.concatenate((cols, rows), axis=-1)

    #compute the concaveHull
    hull = concaveHull(points_2d,5)   #https://github.com/sebastianbeyer/concavehull

    #convert the points into a binary mask (chull)
    polygon = []
    for i in range(np.shape(hull)[0]):
        polygon.append(hull[i][0])
        polygon.append(hull[i][1])

    img = Image.new('L', (100,100), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    chull = np.array(img)

    #resize to the original size
    chull = cv2.resize(chull,(y,x))

    #save the concave hull
    chull = (chull*255.0).astype('uint8')

    #invert the convex hull for post-processing
    chull = 255.0 - chull

    properties = regionprops(label(chull))

    a=[]
    for r in properties:
        a.append(r.area)

    max_ = max(a)

    properties = [a for a in properties if a.area >= 0.05*max_]
    ## remove small regions from the mask as well
    chull_proc = np.zeros(np.shape(chull))
    for r in properties:
        chull_proc[r._label_image==r.label]=1

    #erode the processed chull
    chull_proc = chull_proc.astype('uint8')
    kernel = np.ones((55,55),np.uint8)
    chull_proc = cv2.erode(chull_proc,kernel,iterations = 1)

    #invert again
    chull_proc = 1-chull_proc
    chull_proc = (chull_proc*255.0).astype('uint8')

    #median filter to reduce staircase like borders
    final_chull = signal.medfilt2d(chull_proc, 55)

    final_chull = final_chull/np.max(final_chull)
    final_chull = (final_chull*1).astype('uint8')

    return final_chull

ellipsoid = draw.ellipsoid(9,9,3, spacing=(1,1,1), levelset=False)
ellipsoid = ellipsoid.astype('uint8')
ellipsoid = ellipsoid[1:-1,1:-1,1:-1]

vessel_features = pd.DataFrame(columns=('Image', 'Group', 'Branching Points Density',
                                  'Vessel Density %', 'Avascular Area %', 
                                  'Mean Branch Length', 'Mean Vessel Radius'))

i=0

for msk_name in os.listdir(masks_dir):
    print('Mask Name: {}'.format(msk_name))
    
    #get the resolution information
    aux_res = resolution[resolution['Image'] == msk_name]
    dimx = aux_res['resx'].values[0]
    dimy = aux_res['resy'].values[0]
    dimz = aux_res['resz'].values[0]
    print('dimensions: {} {} {}'.format(dimx, dimy, dimz))

    #mask and region of interest (ROI)
    mask = tifffile.imread(os.path.join(masks_dir, msk_name))
    mask[mask!=0] = 1
    mask = mask.astype('uint8')

    #perform 3D morphological closing operation on the mask
    mask = binary_dilation(mask, ellipsoid)
    mask = binary_erosion(mask, ellipsoid)
    mask = mask*1
    mask = mask.astype('uint8')
    
    print('Mask Shape: {}'.format(np.shape(mask)))

    chull = compute_chull(mask)

    # Perform resampling using cubic interpolation
    mask = ndi.zoom(mask, (dimx, dimy, dimz), order=3, mode='nearest')

    chull = cv2.resize(chull, (np.shape(mask)[1], np.shape(mask)[0]))

    ch3d = convex_hull_image(mask)

    mask_roi = np.zeros(np.shape(mask))
    chull_3d = np.zeros(np.shape(mask))
    for z in range(0, np.shape(mask)[2]):
        chull_3d[:,:,z] = np.logical_and(chull, ch3d[:,:,z])
        mask_roi[:,:,z] = np.logical_and(mask[:,:,z], chull_3d[:,:,z]) #select the region of interest

    mask_roi = (mask_roi*1).astype('uint8')
    chull_3d = (chull_3d*1).astype('uint8')

    print('skeletonization')
    skeleton = skeletonize_3d(mask_roi)
    skeleton = skeleton.astype('uint8')

    #compute the vascular density and avascular area
    mask[chull_3d==0] = 100 #ignore the area outside the ROI
    total_area = len(chull_3d[chull_3d==1]) #total area of the ROI
    vasc_dens = (len(mask[mask==1]) / (total_area) ) * 100 #vascular density
    avas_area = (len(mask[mask==0]) / (total_area) ) * 100 #avascular area

    #features extracted based on the skeleton using the skan package
    branch_data = summarize(Skeleton(skeleton, spacing=[1,1,1]))
    bpoints, bids, epoints, eids = branching_points(branch_data)
    
    print('Skeletons Features Computed')
    mask = mask_roi #select the region of interest
    mask = (mask*1).astype('uint8')

    print('Mask Shape: {}'.format(np.shape(mask)))

    distance_transform = ndi.distance_transform_edt(mask, sampling=[1,1,1]).astype(np.float32)

    print('Distance Transform Computed')
    #distance_transform = distance_transform_edt(mask, sampling=[dimx, dimy, dimz])
    skeleton = skeleton.astype(distance_transform.dtype)
    radius_values = cv2.multiply(skeleton, distance_transform)
    #radius_values = radius_values  #physical units
    radius_ = np.mean(radius_values[radius_values!=0])
    
    #add the features to the pandas dataframe
    vessel_features.loc[i] = [msk_name, msk_name.split('_')[0], bpoints/total_area, 
                                      vasc_dens, avas_area, 
                                     (branch_data['branch-distance']).mean(), radius_]
    i +=1
    
vessel_features.to_csv('features3d.csv', index=False, sep=';')  #um and concave hull do gt, masks cycçlegan 10092022,postprocessed