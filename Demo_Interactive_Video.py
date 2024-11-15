# Interactive Demo of material segmentation net, for video  select point in a singlee and find materials similar to point in sequential frames
# This should display the segmentation map and material similarity map for a selected point in a frame
#  Basically select a point in a given frame and the net will try to find materials similar to this point/frame in the video
# (the segment of the point and how similart the rest of the materials in the rest  of the image to the point.)
# image, segment contour, similarity map, selected point marked blue, Arrows: move dot, +/- change dot size, PageUp/PageDown change threshold
import os.path

import numpy as np
import torch
import cv2
import argparse

import Desnet
print("Arrows control position of selected point"
      "+/- control size of selected point"
      "Enter select point as reference materials (note this selection will be used for all frames until new point will be selected"
      "Page up/Page down  increase decrease threshold"
      "Space go to next frame"
      "m  move to next frame automatically (basically role the video"
      "r start/stop recording meaning every segmented frame will be saved to video"
      "~/1 increase decrease gap between frames, basically make the movie move faster or slower")
##################################################################################33


# Get mask of the selected  point

#######################################################################################
def get_pointer_mask(x,y,radius,im):
    point_mask=np.zeros_like(im[:,:,0],dtype=np.uint8)
    point_mask = cv2.circle(point_mask, (x, y), radius, color=1, thickness=-1)
    return point_mask

##################################################################################3

# Get similarity map from descriptor map and a point (similarity of image to point)


def get_prob_map(fmap,point_mask, desc=None):
    fmap = torch.nn.functional.normalize(fmap, dim=2)
    if desc==None:
        pnt = torch.tensor(point_mask).cuda()
        pnt = pnt.unsqueeze(2)
        pnt = torch.tile(pnt, [1, 1, fmap.shape[2]])

        desc = (pnt * fmap).sum(0).sum(0)  # get sum of descriptors inside the point area

        desc /= (desc ** 2).sum() ** 0.5  # Normalize descriptor
    prob_map = (desc*fmap).sum(2)
    return prob_map.detach().cpu().numpy(),desc
##################################################################################################################3

# Resize image if too big

###############################################################################################3
def resize(im,max_size):
    h,w,d=im.shape
    r=np.min([max_size/h,max_size/w])
    if r<1:
        h=int(h*r)
        w=int(w*r)
        im=cv2.resize(im,(w,h))
    return im




###############################################################################################
# input parameters
parser = argparse.ArgumentParser(description='Apply segmentation, find similarity of materials in the image to a selected point in the image')
parser.add_argument('--video_path', default="/media/deadcrow/6TB/TempVide/mat/Camera/20241107_213139.mp4", type=str, help='video')
parser.add_argument('--out_dir', default="", type=str, help='folder where output will be saved')
#parser.add_argument('--image_path', default="samples/MatSegBenchMark/images_selected/20230913_185027.jpg", type=str, help='target image')
parser.add_argument('--px', default= 300, type=int, help='x cordinate of selected point')
parser.add_argument('--py', default= 350, type=int, help='y cordinate of selected point')
parser.add_argument('--radius', default= 10,type=int, help='radius of selected point')
parser.add_argument('--model_path', default="logs/Defult.torch", type=str, help='trained model path')
parser.add_argument('--desc_len', default= 128, type=int, help='depth of output descriptor')
parser.add_argument('--max_im_size', default= 700, type=int, help='shrink image if too big')
parser.add_argument('--thresh', default= 0.9, type=float, help='threshold to turn soft similarity map to hard segmentation')
args = parser.parse_args()

#---------------------------------------------------------------------------------
out_dir = args.out_dir
if out_dir == "":
    out_dir = args.video_path[:-4]+"//"
if not os.path.exists(out_dir): os.mkdir(out_dir)
for ii in range(10000):
    out_vid = out_dir +"/"+ str(ii)+".mp4"
    if not os.path.exists(out_vid): break
 #------------------------------------------------------------------------------
move=False
record=False
if __name__ == "__main__":
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # load net

    dnet = Desnet.Net(descriptor_depth=args.desc_len)

    dnet.load_state_dict(torch.load(args.model_path))
    dnet.eval()

    # display results
    # Loop through each frame of the video
    desc=None

    ttt=0
    gap=3
    ky=0
    frame=0
    while cap.isOpened():
        # Read the next frame from the video
        ret, im = cap.read()
        ttt+=1
        if ttt%gap>0: continue

        # Check if the frame was read correctly
        if not ret:
            print("End of video or error in reading a frame.")
            break
#---------------------------------------------------------------------------------------------------------------------

        im = resize(im, max_size=args.max_im_size)

        im = np.expand_dims(im, 0)
    # Run net
        with torch.no_grad():
           fmap = dnet.forward(im,im[:,:,:,0]*0, TrainMode=False)
        im=im[0]




        # get point
        while(True):
            img=im.copy()
            point_mask = get_pointer_mask(args.px, args.py, args.radius, img)

            # find similarity of materia at point to the rest of the image
            prob_map,desc = get_prob_map(fmap[0].swapaxes(0, 2).swapaxes(0, 1), point_mask,desc)
            im_cont = img.copy()
            im_overlay = img.copy()

           ## cv2.imshow("prob_map", ((prob_map) / prob_map.max() * 255).astype(np.uint8))

            contour=(prob_map > args.thresh).astype(np.uint8)-cv2.erode((prob_map > args.thresh).astype(np.uint8),np.ones([5,5])) # segment contour

            # place contour on image
            im_cont[:, :, 0][contour > 0] = 0
            im_cont[:, :, 1][contour > 0] = 255
            im_cont[:, :, 2][contour > 0] = 0

            # place overlay on image
            im_overlay[:, :, 1][prob_map > args.thresh] =  0#(im_overlay[:, :, 0]/ (1.0 + 2*(prob_map > args.thresh).astype(np.float32))).astype(np.uint8)
         #   im_overlay[:, :, 2][prob_map > args.thresh] = (im_overlay[:, :, 0]* (1.0 + 1*(prob_map > args.thresh).astype(np.float32))).astype(np.uint8)
            prob_map = cv2.cvtColor((prob_map * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            ####################################
            comb_im=np.hstack([img, im_overlay, im_cont, prob_map])
            if record:
                if frame == 0:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4 files
                    video_writer = cv2.VideoWriter(out_vid, fourcc, 23, (comb_im.shape[1], comb_im.shape[0]))
                    # place point on image
                    img[:, :, 0][point_mask > 0] = 255
                    img[:, :, 1][point_mask > 0] = 255
                    img[:, :, 2][point_mask > 0] = 0

                    # place point on image
                    im_cont[:, :, 0][point_mask > 0] = 255
                    im_cont[:, :, 1][point_mask > 0] = 255
                    im_cont[:, :, 2][point_mask > 0] = 0
                frame+=1
                video_writer.write(comb_im)
                print("write",frame, " path ", out_vid)

            ##############################################



            # place point on image
            img[:, :, 0][point_mask > 0] = 255
            img[:, :, 1][point_mask > 0] = 255
            img[:, :, 2][point_mask > 0] = 0

            # place point on image
            im_cont[:, :, 0][point_mask > 0] = 255
            im_cont[:, :, 1][point_mask > 0] = 255
            im_cont[:, :, 2][point_mask > 0] = 0




          ###  cv2.imshow("prob_map", ((prob_map )  * 255).astype(np.uint8))

            cv2.imshow("image, segment contour, similarity map, selected point marked blue, Arrows: move dot, +/- change dot size, PageUp/Down change threshold", np.hstack([img,im_overlay,im_cont,prob_map]));
            if not move:
                ky = cv2.waitKey()
            else:
                ky = cv2.waitKey(100)
            print(ky)

            # Arrows
            if ky == 109: # m
                move= not move # automatically jump to next frame even where no key is pressed
            if ky == 81: args.px -= 5 # arrows
            if ky == 83: args.px += 5
            if ky == 82: args.py -= 5
            if ky == 84: args.py += 5
            if ky == 43:  args.radius += 2 # +
            if ky == 45:  args.radius -= 2# -
            if ky == 85:  args.thresh += 0.02 # pagup
            if ky == 86:  args.thresh -= 0.02# pagedown
            if ky == 32 or ky==27 or move:  break # space is next frame esc to exis
            if ky == 13: desc=None
            if ky == 49: gap+=1 # 1
            if ky == 96: gap-=1# ~
            if ky == 114: record = not record  # r # record


            if gap == 0: gap=1
            print("gap=",gap,"   record",record, "threshold", args.thresh)

            if args.radius<1: args.radius=1
        if ky==27: break
    video_writer.release()

    # Left: 37
    # Up: 38
    # Right: 39
    # Down: 40