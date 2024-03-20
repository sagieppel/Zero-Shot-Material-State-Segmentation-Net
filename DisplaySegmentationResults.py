import os
import sys
import numpy as np
import cv2
import json
import pickle
import hashlib
import shutil
import os
import EvaluateWithSoft
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse

import Desnet



#######################################################################################################################################


##################################################################################33


# Get random points inside mask

#######################################################################################
def get_pointer_masks(mask0,num_points,boundary_erosion,point_radius,min_erode_area):
    mask=mask0.copy().astype(np.uint8)
    erode_mask=cv2.erode(mask,np.ones([boundary_erosion+point_radius,boundary_erosion+point_radius],dtype=np.uint8))
    if erode_mask.sum()<min_erode_area:
               return False,False,False


    pointer_mask_list=[]
    pointer_positions=[]
    for ip in range(num_points):
        while(True):
           y=np.random.randint(mask.shape[0])
           x=np.random.randint(mask.shape[1])
           if erode_mask[y,x]>0: break
        point_mask=np.zeros_like(mask,dtype=np.uint8)
        point_mask = cv2.circle(point_mask, (x, y), point_radius, color=1, thickness=-1)
        pointer_mask_list.append(point_mask)
        pointer_positions.append([x,y,point_radius])

    # for pmsk in  pointer_mask_list:
    #      msk = mask0.copy()
    #      msk[mask>0]=125
    #      msk[pmsk>0]=255
    #      cv2.imshow("",msk)
    #      cv2.waitKey()
    return True,pointer_mask_list,pointer_positions

##################################################################################3

def get_prob_map(img,fmap,point_mask,rand=False):
    fmap = torch.nn.functional.normalize(fmap, dim=2)
    pnt = torch.tensor(point_mask).cuda()
    pnt = pnt.unsqueeze(2)
    pnt = torch.tile(pnt, [1, 1, fmap.shape[2]])

    desc = (pnt * fmap).sum(0).sum(0)  # get sum of descriptors inside the point area
    if rand == True:
        desc = torch.rand(desc.shape)
    desc /= (desc ** 2).sum() ** 0.5  # Normalize descriptor
    prob_map = (desc*fmap).sum(2)
    desc = desc.cpu().numpy()
    return prob_map.cpu().numpy()







###############################################################################################

if __name__ == "__main__":
    model_path = "logs/Defult.torch"  # 94.3 95.0  (700)
    image_path = "samples/MatSegBenchMark/images_selected/20230901_183620.jpg"
    thresh=0.5
    map_type="logit"
    im = cv2.imread(image_path)

    out_dir=r"/mnt/306deddd-38b0-4adc-b1ea-dcd5efc989f3/MatSegPaperImage/Figure7/RawData2/"
    prob_dir=out_dir+"/prob_map/"
    overlay_dir=out_dir+"/overlay/"
    img_dir = out_dir + "/img/"
    concat_dir = out_dir + "/concat/"
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    if not os.path.exists(prob_dir): os.mkdir(prob_dir)
    if not os.path.exists(overlay_dir): os.mkdir(overlay_dir)
    if not os.path.exists(img_dir): os.mkdir(img_dir)
    if not os.path.exists(concat_dir): os.mkdir(concat_dir)
    # images_dir = "/home/breakeroftime/Desktop/MatSegBenchMarkSelected/"#/home/breakeroftime/Desktop/MatSegBenchMarkSelected/"
    # data_dir = "/home/breakeroftime/Desktop/MatSegBenchMarkSelected/data/"
    #____________________DMS______________________________________________________________
    # DMSlabel_dir = "/media/breakeroftime/2T/Data_zoo/dms_v1_labels/DMS_v1/Labels_Masks/test/"
    # DMSimage_dir = "/media/breakeroftime/2T/Data_zoo/dms_v1_labels/images/test/"
    # reader = READER_DMS.Reader(DMSimage_dir, DMSlabel_dir,max_case_per_cat=50)
    #_________________LabPics________________________________________________
    # LabPicsDir = r"/mnt/306deddd-38b0-4adc-b1ea-dcd5efc989f3/Data_zoo/LabPics2.5/Chemistry/Test/"
    # reader = READER_LABPICS.Reader(LabPicsDir)




    #--------------Materialistic_______________________________________________
    # image_dir = "/mnt/306deddd-38b0-4adc-b1ea-dcd5efc989f3/Data_zoo/materialistic_real_data/images/"
    # labels_dir = "/mnt/306deddd-38b0-4adc-b1ea-dcd5efc989f3/Data_zoo/materialistic_real_data/masks/"
    # reader = READER_MATERIALISTIC.Reader(image_dir, labels_dir)
    #-------------------MatSeg--------------------------------------------------
    reader = EvaluateWithSoft.TestSetReader(images_dir=images_dir, data_dir=data_dir, max_size=800)

    #-------------------------------------------------------------------------------
 #   while (True):
 #       img, mask, cc_mask, roi = reader.read_next(max_size=500, min_size=200)
    desc_length = 128
 #   max_size = 700 #700 # worked good
    dnet = Desnet.Net(descriptor_depth=desc_length)

    dnet.load_state_dict(torch.load(model_path))
    dnet.cuda()
    dnet.eval()

    union_lst = []
    intersection_lst = []
    iou_lst = []
    reader.itr = 1
   # for itr in range(80020):
    for iii in range(10000000):
        img, matLdt, width, height, matmasks = reader.load(only_read_with_soft_relations=False)
        gtmask=np.ones(img.shape[:2])

        #img, gtmask, cc_mask, roi = reader.read_next(max_size=900, min_size=200)
        #gtmask,cc_mask,roi = (gtmask>0).astype(np.uint8), (cc_mask>0).astype(np.uint8), (roi>0).astype(np.uint8)
        #if gtmask.sum()<900: continue
        #    img,matLdt ,width, height, matmasks = reader.load()
       # print("size",img.shape,"sum:",img.sum())
        # ---------Main traininf loop----------------------------------------------------------------------------------------------------------------------------
        print("------------------------------------",reader.itr,"------------------------")
       # if itr<405: continue
        with torch.no_grad():
                img = np.expand_dims(img, 0)
                fmap = dnet.forward(img,img[:,:,:,0]*0, TrainMode=False)
            #****************************************************************************************************************888
             #   thresh,map_type=SegmentFromFeatureMap.single_segment_from_GT_points(fmap[0].swapaxes(0, 2).swapaxes(0, 1),img,matmasks,thresh,map_type)#******************************************
              #  SegmentFromFeatureMap.full_segmention_from_GT_points(fmap[0].swapaxes(0, 2).swapaxes(0, 1),img,matmasks)
           #********************************************************************************************************
                success,pointer_mask_list,pointer_positions = get_pointer_masks(gtmask, num_points=10,boundary_erosion=10, point_radius=10, min_erode_area=400)
               # success, pointer_mask_list, pointer_positions = get_pointer_masks(gtmask, num_points=10,boundary_erosion=0, point_radius=10,min_erode_area=400)
                if not success: continue
               ### for n,pmask in enumerate(pointer_mask_list):
                for ky in matmasks:
                    for ky2 in matmasks[ky]:
                        pmask=matmasks[ky][ky2]
                        prob_map = get_prob_map(img,fmap[0].swapaxes(0, 2).swapaxes(0, 1),pmask)
                        for thresh in [0]:

                            Fprob_map = prob_map.copy()
                            Fprob_map[Fprob_map<0]=0
                            Fprob_map[Fprob_map < thresh] = 0
                            Fprob_map-=Fprob_map.min()
                            prob_map_view = (Fprob_map/Fprob_map.max()*255).astype(np.uint8)


                            im=img[0].copy()
                            im_marked=im.copy()
                          #  cv2.imshow("img"+str(thresh), im)
                            im_marked= im_marked.astype(np.float32)*0.70
                            #im_marked[:, :, 2] = (im_marked[:, :, 2].astype(np.float32) + prob_map_view.astype(np.float32) /2).astype(np.uint8) #

                            im_marked[:,:,2] = prob_map_view#(im[:,:,2].astype(np.float32)/4 + prob_map_view.astype(np.float32)*3/4).astype(np.uint8)

                            im[:, :, 0][pmask > 0] = 255
                            im[:, :, 1][pmask > 0] = 0
                            im[:, :, 2][pmask > 0] = 0

                            im_marked[:, :, 0][pmask > 0] = 255
                            im_marked[:, :, 1][pmask > 0] = 0
                            im_marked[:, :, 2][pmask > 0] = 0

                            prob_map_view = np.stack((prob_map_view,) * 3, axis=-1)

                            prob_map_view[:, :, 0][pmask > 0] = 255
                            prob_map_view[:, :, 1][pmask > 0] = 0
                            prob_map_view[:, :, 2][pmask > 0] = 0
                            img_name = reader.img_name[:-4]+"_mat_"+ky+"_point_"+str(ky2)+".png"
                            cv2.imwrite(prob_dir + "/" + img_name, prob_map_view)
                            cv2.imwrite(overlay_dir + "/" + img_name, im_marked)
                            cv2.imwrite(img_dir + "/" + img_name, img)
                            cv2.imwrite(concat_dir + "/" + img_name, np.hstack([im,im_marked,prob_map_view]))
                            print("write to ",concat_dir + "/" + img_name)

                            # cv2.imshow("prob_map", prob_map_view)
                            # im = img[0].copy()
                            #
                            # cv2.imwrtie(prob_dir + "/" + reader.img_name, prob_map_view)
                            # cv2.imshow("img marked" + str(thresh), im)
                            # cv2.waitKey()
                            # cv2.destroyAllWindows()
     #
#                         bst_iou,bst_thresh,prd_mask= find_best_tresh_and_IOU(prob_map, gtmask>0,roi,min_thresh=0,mx_thresh=1,gap=0.05,min_gap=0.01)
#                         # cv2.imshow(str(bst_iou), (prd_mask*2+gtmask).astype(np.uint8)*60)
#                         # cv2.waitKey()
#                         # cv2.destroyAllWindows()
#                         iou_lst.append(bst_iou)
#                 print(len(iou_lst)/10,iii,"Mean IOU=",np.array(iou_lst).mean(),"   IOU=",bst_iou)
#
#
#
#
#
# ###############################3
# #############################
# ########################
# ##############################
# if __name__ == "__main__":
#     thresh=0.5
#     map_type="logit"
#     # Great Results
#     model_path = "DescNet_LOGS_Convnext_mode2_4_Temp02_Desc128_Work_Great/Defult.torch"# 94.3 95.0  (700)
#     images_dir = "/home/breakeroftime/Desktop/MatSegBenchMarkSelected/"#/home/breakeroftime/Desktop/MatSegBenchMarkSelected/"
#     data_dir = "/home/breakeroftime/Desktop/MatSegBenchMarkSelected/data/"
#
#     #
#
#     #first: 0.9422368854154407
#     #first: 0.9493024692108405
#
#     #
#     desc_length = 128
#     max_size = 700 #700 # worked good
#     dnet = Desnet.Net(descriptor_depth=desc_length)
#
#     dnet.load_state_dict(torch.load(model_path))
#     dnet.cuda()
#     dnet.eval()
#
#     accuracy_point_list = []
#     accuracy_mat_list = []
#     accuracy_point_first_list = []
#     accuracy_mat_first_list = []
#     reader = EvaluateWithSoft.TestSetReader(images_dir=images_dir, data_dir=data_dir,max_size=max_size) #900
#     reader.itr = 1
#    # for itr in range(80020):
#     while (True):
#         img,matLdt ,width, height, matmasks = reader.load(only_read_with_soft_relations=False)
#         print("size",img.shape,"sum:",img.sum())
#         # ---------Main traininf loop----------------------------------------------------------------------------------------------------------------------------
#         print("------------------------------------",reader.itr,"------------------------")
#        # if itr<405: continue
#         with torch.no_grad():
#                 img = np.expand_dims(img, 0)
#                 fmap = dnet.forward(img,img[:,:,:,0]*0, TrainMode=False)