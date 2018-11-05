import numpy as np
def MultiInstance_patch(images, landmarks, patch_size, batch_size,numofscales):
    paddingsize =24
    landmarks = np.round(landmarks)+paddingsize
    landmk_num = np.size(landmarks,2)
    patches = np.zeros((batch_size,landmk_num,numofscales,patch_size,patch_size,patch_size),dtype='uint8')

    while True:
        i_sample = 0

        for i_subject in range(np.size(images,0)):
            image_subject=np.lib.pad(images[i_subject,...], ((paddingsize,paddingsize),(paddingsize,paddingsize),(paddingsize,paddingsize)), 'constant', constant_values=0)
            for i_landmk in range(0,landmk_num):

                i_x = np.random.permutation(1)[0] + landmarks[i_subject, 0, i_landmk]
                i_y = np.random.permutation(1)[0] + landmarks[i_subject, 1, i_landmk]
                i_z = np.random.permutation(1)[0] + landmarks[i_subject, 2, i_landmk]
                for i_scale in range(numofscales):
                    patches[i_sample,i_landmk,i_scale,0:patch_size,0:patch_size,0:patch_size] = image_subject[int(i_x-np.floor(patch_size*(i_scale+1)/2)):int(i_x+int(np.ceil(patch_size*(i_scale+1)/2.0))):i_scale+1,
                                                                                           int(i_y-np.floor(patch_size*(i_scale+1)/2)):int(i_y+int(np.ceil(patch_size*(i_scale+1)/2.0))):i_scale+1,
                                                                                           int(i_z-np.floor(patch_size*(i_scale+1)/2)):int(i_z+int(np.ceil(patch_size*(i_scale+1)/2.0))):i_scale+1]


            i_sample +=1

            if i_sample==batch_size:

                mi_data = []
                for j_landmk in range(0,landmk_num):

                    mi_data.append(patches[:,j_landmk,...])

                return mi_data