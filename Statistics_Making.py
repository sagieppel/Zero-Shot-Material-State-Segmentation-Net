# Make statics and save samples for tracing training progress
import numpy as np
import os
import cv2
import random
#############################################################################################
colormap = np.array([
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (0, 255, 255),    # Cyan
    (255, 0, 255),    # Magenta
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Purple
    (255, 255, 255),  # White
    (128, 128, 128),  # Gray
 ], dtype=np.uint8)
#############################################################################################
def generate_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)
#######################################################################################################
# Torch to numpy
def tonumpy(x):
   if x is None:
        return None
   if  'torch' in str(x.dtype):
       return x.cpu().detach().numpy()
   return x

####################################################################################################################
# Turn instancem map to rgb for display

def map_segmentation_to_rgb(segmentation, image, ROI):
    # Define the colormap

    color_mask=np.zeros_like(image)
    color_mask[:,:,0][ROI>0] = 255
    # Apply colormap to segmentation map
    for i in np.unique(segmentation):
         if i==0: continue
         for k in range(3):
            if i<len(colormap):
               color_mask[:,:,k][segmentation==i]=colormap[int(i)][k]
            else:
                color_mask[:, :, k][segmentation == i] = random.randint(0, 255)
    color_mask[ROI<0.1]=0
    return color_mask
############################################################################################################################################
# Display one hot of gt and prediction

def display_one_hot(Imgs,one_hot1,one_hot2=None,ROI=None,outdir=None,disp=False,save=True):
    if not os.path.exists(outdir) and save: os.mkdir(outdir)
    Imgs=tonumpy(Imgs)
    one_hot1 = tonumpy(one_hot1)
    one_hot2 = tonumpy(one_hot2)
    ROI = tonumpy(ROI)

    for ib in range(one_hot1.shape[0]):
        im = Imgs[ib].copy()
        im=cv2.resize(im,(ROI.shape[2],ROI.shape[1]))
        for ii in range(one_hot1.shape[1]):
                i = im.copy()
                msk = np.zeros_like(im)
                if (one_hot2[ib][ii]>0.5).sum()<10: continue
                msk[:, :, 0] = (one_hot1[ib][ii]*255).astype(np.uint8)
                im1 =im.copy()
                im1[:, :, 0] = (one_hot1[ib][ii] * 255).astype(np.uint8)*ROI[ib]
                im1[:, :, 1] = (one_hot1[ib][ii] * 255).astype(np.uint8)*ROI[ib]
                im1[:, :, 2] = (one_hot1[ib][ii] * 255).astype(np.uint8)*ROI[ib]
                if one_hot2 is not None:
                    msk[:, :, 1] = (one_hot2[ib][ii]*255).astype(np.uint8)*ROI[ib]
                    im2 = im.copy()
                    im2[:, :, 0] = (one_hot2[ib][ii]*255).astype(np.uint8)*ROI[ib]
                    im2[:, :, 1] = (one_hot2[ib][ii] * 255).astype(np.uint8)*ROI[ib]
                    im2[:, :, 2] = (one_hot2[ib][ii] * 255).astype(np.uint8)*ROI[ib]
                if ROI is not None:
                    msk[:, :, 2] = (ROI[ib] * 255).astype(np.uint8)
                over1 = np.hstack([i,msk])
                over2 = np.hstack([i, im1,im2])
                if save:
                    cv2.imwrite(outdir + "/" + str(ib) + "Msk" + str(ii)+'.jpg', over1.astype(np.uint8))
                    cv2.imwrite(outdir + "/" + str(ib) + "Msk" + str(ii)+"b.jpg", over2.astype(np.uint8))
                    if ROI is not None: cv2.imwrite(outdir + "/" + str(ib) + "_ROI.jpg", (ROI[ib]*255).astype(np.uint8))
                if disp:
                    cv2.destroyAllWindows()
                    cv2.imshow(outdir + "/" + str(ib) + "Msk" + str(ii), over1.astype(np.uint8))
                    cv2.imshow(outdir + "/" + str(ib) + "Msk" + str(ii)+ "b", over2.astype(np.uint8))
                    if ROI is not None: cv2.imwrite(outdir + "/" + str(ib) + "_ROI.jpg",(ROI[ib]*255).astype(np.uint8))
                    cv2.waitKey()
                    print("d")

################################################################################################################################################
# Get Intersection over union of GT and predicted  region (within the ROI region)

def get_one_hot_IOU(one_hotpr,one_hotgt,ROI):
    IOU = 0
    pxIOU = 0
    GTSum = 0
    n = 0
    for ib in range(one_hotgt.shape[0]):

        for ii in range(one_hotgt.shape[1]):
            inter  = ((one_hotgt[ib][ii]>0.5) * (one_hotpr[ib][ii]>0.5) *ROI[ib]).sum()
            gtsum = (one_hotgt[ib][ii]>0.5).sum()
            prdsum = (one_hotpr[ib][ii]>0.5).sum()

            union = gtsum + prdsum -inter + 0.00001
            #if union<100: continue
            iou = inter/(union+0.001)
            IOU += iou
            pxIOU += iou*gtsum
            GTSum += gtsum
            n+=1
    if n==0 or GTSum==0:
        return 0,0

    return IOU/n, pxIOU/GTSum



######################################################################################################################################################

############################################################################################################################################
# Display two segmentation map (by turning to RGB) for displaying GT segmentation
def display_sgmap(Imgs,sgmap1,sgmap2=None,ROI=None,outdir=None,disp=False,save=True,iou=None):
    if not os.path.exists(outdir) and save: os.mkdir(outdir)

    for ib in range(Imgs.shape[0]):
        im = Imgs[ib].copy()
        im = cv2.resize(im, (ROI.shape[2], ROI.shape[1]))
        colored_mask= map_segmentation_to_rgb(sgmap1[ib], im, ROI[ib])
        if sgmap2 is not None:
            colored_mask2 = map_segmentation_to_rgb(sgmap2[ib], im, ROI[ib])
            colored_mask = np.hstack([colored_mask2, colored_mask,im.astype(np.uint8)])
        if iou is not None:
            txt = "_iou" + str(iou) + ".jpg"
        else:
            txt = ".jpg"
        if save:
            cv2.imwrite(outdir+"/"+str(ib)+"_color"+txt,colored_mask.astype(np.uint8))
            cv2.imwrite(outdir + "/" + str(ib) + "_overlay"+txt, im.astype(np.uint8))
            if ROI is not None: cv2.imwrite(outdir + "/" + str(ib) + "_ROI.jpg", (ROI[ib]*255).astype(np.uint8))
        if disp:
            cv2.imshow("segmentation results predicted vs GT"+txt, colored_mask)
            cv2.imshow(outdir + "/" + str(ib) + "_overlay"+txt, Imgs[ib].astype(np.uint8))
            if ROI is not None: cv2.imshow(outdir + "/" + str(ib) + "_ROI.jpg", (ROI[ib]*255).astype(np.uint8))

            cv2.waitKey()
            cv2.destroyAllWindows()
###########################################################################################################################################
# Given batch of segmentatiion maps (prediction/gt) find the IOU between them (in the ROI)
def IOU_sgmap(pred,gt,ROI):
    IOU = 0 # IOU image average
    pxIOU = 0 # IOU pixel average
    GTSum = 0
    n = 0
    iou_list=[]
    for ib in range(gt.shape[0]):
        tmp_iou=[]
        for i in np.unique(gt[ib]):  # exclude background
            #if i==0: continue
            intersection = ((pred[ib] == i) *  (gt[ib] == i)*ROI[ib]).sum()
            union = (pred[ib] == i).sum() + (gt[ib] == i).sum() - intersection
            if union==0: continue
            iou = intersection / union if union > 0 else 0
            tmp_iou.append(iou)
            IOU+=iou
            pxIOU+=iou*(gt == i).sum()
            GTSum+=(gt == i).sum()
            n+=1
        iou_list.append(np.mean(tmp_iou))
    return IOU / n, pxIOU / GTSum,iou_list


###############################################################################################################
# Get statics IOU on the batch of predicted segmentation map and GT segmentation map withing ROI
# Also save samples of the results in the log dir
def get_statics(Imgs, prd_seg_map, gt_seg_map, gt_one_hot,logdir,itr,save,display,ROI):#prd_prob,
    Imgs = tonumpy(Imgs)# to numpy
   # prd_prob = tonumpy(prd_prob)
    prd_seg_map = tonumpy(prd_seg_map)
    gt_seg_map = tonumpy(gt_seg_map)
    gt_one_hot = tonumpy(gt_one_hot)
    ROI = tonumpy(ROI)
    if not os.path.exists(logdir):os.mkdir(logdir)
    iou, iou_area, iou_list = IOU_sgmap(prd_seg_map, gt_seg_map, ROI) # get IOU
  #  iou_o, iou_area_o = get_one_hot_IOU(prd_prob, gt_one_hot, ROI)
    if  save or display: # for debuging display or save predicted and GT segmentation maps next toeach other
          outdir=logdir+"/"+str(itr)+"/"
          if save and not os.path.exists(outdir): os.mkdir(outdir)
          if  save or display: # for debuging display or save predicted and GT segmentation maps next toeach other
              #display_one_hot(Imgs, prd_prob, gt_one_hot, outdir=outdir+"/one_hot2/", disp=display, save=save, ROI=ROI)
              display_sgmap(Imgs, prd_seg_map, gt_seg_map, outdir=outdir+"/segmap2/", disp=display, save=save,ROI=ROI,iou=iou_list)


    #print("IOU=",  iou,"---",iou_o,"IOU By Area =", iou_area,"---",iou_area_o)
    return   iou,iou_area,