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
from scipy.sparse.csgraph import shortest_path
from skan.csr import skeleton_to_csgraph

masks_dir = r'/mnt/2TBData/hemaxi/cycleGAN/26_05_2022/models_01_07/masks_proc'
resolution_file = r'/mnt/2TBData/hemaxi/Downloads/resolution.xlsx'

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


def radius_3d_main(skeleton, branch_data, mask, boundary):
    pixel_graph, coordinates_ = skeleton_to_csgraph(skeleton, spacing=[1,1,1])
    dist_matrix, predecessors = shortest_path(pixel_graph, directed=True, indices=branch_data['node-id-src'].to_numpy(), return_predecessors=True)
    #dist_matrix has size (#sourceids as in len(branch_data['node-id-src']), #all nodes in the skeleton)
    
    ## iterate through each branch and check the direction
    all_major = []
    all_minor = []
    all_radii = []

    for i in range(len(branch_data)):
        
        #node indices (i is the node-id-src, because it was used before to compute dist_matrix and predecessors)
        b =  int(branch_data.iloc[i]['node-id-dst']) 

        # Check if there is a path between the two nodes (a and b)
        if np.isinf(dist_matrix[i, b]):
            print("No path exists between node a and node b.")
            continue
        else:
            # Reconstruct the path from a to b
            path = [(coordinates_[0][b], coordinates_[1][b], coordinates_[2][b])]
            b = predecessors[i, b]
            while b >= 0:
                path.insert(0, (coordinates_[0][b], coordinates_[1][b], coordinates_[2][b]))
                b = predecessors[i, b]

            path = np.asarray(path)
            #print("Shortest path:", path)        

            #compute the direction of the branch
            delta_x = (branch_data.iloc[i]['image-coord-src-0'])-(branch_data.iloc[i]['image-coord-dst-0'])
            delta_y = (branch_data.iloc[i]['image-coord-src-1'])-(branch_data.iloc[i]['image-coord-dst-1'])
            delta_z = (branch_data.iloc[i]['image-coord-src-2'])-(branch_data.iloc[i]['image-coord-dst-2'])

            direction_unit = np.asarray([delta_x, delta_y, delta_z])
            direction_unit = direction_unit / np.linalg.norm(direction_unit)

            major_axes, minor_axes, radii = compute_radii_aux(path, mask, boundary, direction_unit)

            all_major = all_major + major_axes
            all_minor = all_minor + minor_axes
            all_radii = all_radii + radii
                
    return np.asarray(all_major), np.asarray(all_minor), np.asarray(all_radii)

def extract_2d_slice(segmentation_mask, boundary, point, direction_unit, radius=20):
    D = np.dot(direction_unit, point)
    min_point = np.maximum(point - radius, [0, 0, 0])
    max_point = np.minimum(point + radius, segmentation_mask.shape)
    
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(min_point[0], max_point[0]),
        np.arange(min_point[1], max_point[1]),
        np.arange(min_point[2], max_point[2]),
        indexing='ij')
    
    voxel_centers = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
    distances = np.abs(np.dot(voxel_centers, direction_unit) - D)
    plane = (distances < 0.5).reshape(grid_x.shape)
    
    out_mask = np.zeros(plane.shape)
    out_mask = np.logical_and(plane, segmentation_mask[min_point[0]:max_point[0], min_point[1]:max_point[1], min_point[2]:max_point[2]])
    
    out_boundary = np.zeros(boundary.shape)
    out_boundary = np.logical_and(plane, boundary[min_point[0]:max_point[0], min_point[1]:max_point[1], min_point[2]:max_point[2]])
    
    new_point = point * (min_point == 0) + radius * (min_point != 0)

    #tifffile.imwrite('testingggggg.tif', (plane*255.0).astype('uint8'))
    #input('press enter')
    return out_boundary, out_mask, new_point

def compute_radii_aux(path, mask, boundary, direction_unit):
    major_axes = []
    minor_axes = []
    radii = []

    tam_ = np.shape(path)[0]

    #positions_ = (np.asarray([1/2, 1/3, 2/3, 1/4, 3/4])*tam_).astype('uint8')
    first_point = True
    for p in range(int(tam_/2), tam_):

        point = np.asarray(path[p])
        #print(point)

        aux_boundary, aux_mask, point = extract_2d_slice(mask, boundary, point, direction_unit)

        aux_mask = label(aux_mask)
        
        l = aux_mask[point[0], point[1], point[2]]
        
        aux_boundary[aux_mask!=l] = 0
        
        #tifffile.imwrite('auxbound.tif', (aux_boundary*255.0).astype('uint8'))
        #tifffile.imwrite('auxmask.tif', (aux_mask*255.0).astype('uint8'))

        #input('Press enter')
        
        indices_ = np.argwhere(aux_boundary) # get indices of the contour

        if np.shape(indices_)[0]>0:

            all_distances = np.sqrt(np.sum((indices_ - point)**2, axis=-1)) #Euclidean distance
            #from the point to each point in the boundary

            major_curr = np.max(all_distances)
            minor_curr = np.min(all_distances)
            
            if first_point:
                major_axes.append(major_curr)
                minor_axes.append(minor_curr)
                radii.append(np.mean(all_distances))
                first_point = False
            else:
                delta_radius_major = np.abs(major_curr-major_axes[-1])
                delta_radius_minor = np.abs(minor_curr-minor_axes[-1])
                if delta_radius_major<4:
                    major_axes.append(major_curr)
                    minor_axes.append(minor_curr)
                    radii.append(np.mean(all_distances))
                else:
                    break
            
    path = np.flip(path,0)
    for p in range(int(tam_/2), tam_):

        point = np.asarray(path[p])
        #print(point)

        aux_boundary, aux_mask, point = extract_2d_slice(mask, boundary, point, direction_unit)

        aux_mask = label(aux_mask)
        
        l = aux_mask[point[0], point[1], point[2]]
        
        aux_boundary[aux_mask!=l] = 0
        
        #tifffile.imwrite('auxbound.tif', (aux_boundary*255.0).astype('uint8'))
        #tifffile.imwrite('auxmask.tif', (aux_mask*255.0).astype('uint8'))

        #input('Press enter')
        
        indices_ = np.argwhere(aux_boundary) # get indices of the contour

        if np.shape(indices_)[0]>0:

            all_distances = np.sqrt(np.sum((indices_ - point)**2, axis=-1)) #Euclidean distance
            #from the point to each point in the boundary
            
            major_curr = np.max(all_distances)
            minor_curr = np.min(all_distances)
            
            if first_point:
                major_axes.append(major_curr)
                minor_axes.append(minor_curr)
                radii.append(np.mean(all_distances))
                first_point = False
            else:
                delta_radius_major = np.abs(major_curr-major_axes[-1])
                delta_radius_minor = np.abs(minor_curr-minor_axes[-1])
                if delta_radius_major<4:
                    major_axes.append(major_curr)
                    minor_axes.append(minor_curr)
                    radii.append(np.mean(all_distances))
                else:
                    break

    return major_axes, minor_axes, radii

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
    mask[chull_3d==0] = 0
    mask = (mask*1).astype('uint8')

    ellipsoid = draw.ellipsoid(1,1,1, spacing=(1,1,1), levelset=False)
    ellipsoid = ellipsoid.astype('uint8')

    er = binary_erosion(mask, ellipsoid)
    boundaries = mask - er

    major_axes_final, minor_axes_final, radii_final = radius_3d_main(skeleton, branch_data, mask, boundaries)
    
    #add the features to the pandas dataframe
    vessel_features.loc[i] = [msk_name, msk_name.split('_')[0], bpoints/total_area, 
                                      vasc_dens, avas_area, 
                                     (branch_data['branch-distance']).mean(), np.mean(radii_final)]
    i +=1
    
vessel_features.to_csv('features3d.csv', index=False, sep=';')  #um and concave hull do gt, masks cycçlegan 10092022,postprocessed