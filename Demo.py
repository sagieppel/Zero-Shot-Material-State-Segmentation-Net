# Demo of material segmentation net, select point in the image and display the similarity of
# This should display the segmentation map and material similarity map for a selected point in the image (the segment of the point and how similart the rest of the materials in the rest  of the image to the point.)

import numpy as np
import torch
import cv2
import argparse

import Desnet

##################################################################################33


# Get mask of the selected  point

#######################################################################################
def get_pointer_mask(x,y,radius,im):
    point_mask=np.zeros_like(im[:,:,0],dtype=np.uint8)
    point_mask = cv2.circle(point_mask, (x, y), radius, color=1, thickness=-1)
    return point_mask

##################################################################################3

# Get similarity map from descriptor map and a point (similarity of image to point)


def get_prob_map(fmap,point_mask,rand=False):
    fmap = torch.nn.functional.normalize(fmap, dim=2)
    pnt = torch.tensor(point_mask).cuda()
    pnt = pnt.unsqueeze(2)
    pnt = torch.tile(pnt, [1, 1, fmap.shape[2]])

    desc = (pnt * fmap).sum(0).sum(0)  # get sum of descriptors inside the point area
    if rand == True:
        desc = torch.rand(desc.shape)
    desc /= (desc ** 2).sum() ** 0.5  # Normalize descriptor
    prob_map = (desc*fmap).sum(2)
    return prob_map.detach().cpu().numpy()
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
parser.add_argument('--image_path', default="samples/MatSegBenchMark/images/20231018_110246.jpg", type=str, help='target image')
#parser.add_argument('--image_path', default="samples/MatSegBenchMark/images_selected/20230913_185027.jpg", type=str, help='target image')
parser.add_argument('--px', default= 300, type=int, help='x cordinate of selected point')
parser.add_argument('--py', default= 350, type=int, help='y cordinate of selected point')
parser.add_argument('--radius', default= 10,type=int, help='radius of selected point')
parser.add_argument('--model_path', default="logs/Defult.torch", type=str, help='trained model path')
parser.add_argument('--desc_len', default= 128, type=int, help='depth of output descriptor')
parser.add_argument('--max_im_size', default= 700, type=int, help='shrink image if too big')
parser.add_argument('--thresh', default= 0.9, type=float, help='threshold to turn soft similarity map to hard segmentation')
args = parser.parse_args()

if __name__ == "__main__":



# load net

    dnet = Desnet.Net(descriptor_depth=args.desc_len)

    dnet.load_state_dict(torch.load(args.model_path))
    dnet.eval()

# load image
    img = cv2.imread(args.image_path)
    img = resize(img, max_size=args.max_im_size)

    img = np.expand_dims(img, 0)
# Run net
    fmap = dnet.forward(img,img[:,:,:,0]*0, TrainMode=False)
    img=img[0]

# get point
    point_mask = get_pointer_mask(args.px,args.py,args.radius,img)

# find similarity of materia at point to the rest of the image
    prob_map = get_prob_map(fmap[0].swapaxes(0, 2).swapaxes(0, 1),point_mask)

# display results

    im = img.copy()

    cv2.imshow("prob_map", ((prob_map) / prob_map.max() * 255).astype(np.uint8))

    contour=(prob_map > args.thresh).astype(np.uint8)-cv2.erode((prob_map > args.thresh).astype(np.uint8),np.ones([5,5])) # segment contour
    # place point on image
    im[:, :, 0][point_mask > 0] = 255
    im[:, :, 1][point_mask > 0] = 255
    im[:, :, 2][point_mask > 0] = 0

    # place contour on image
    im[:, :, 0][contour > 0] = 0
    im[:, :, 1][contour > 0] = 0
    im[:, :, 2][contour > 0] = 255

  ###  cv2.imshow("prob_map", ((prob_map )  * 255).astype(np.uint8))
    prob_map = cv2.cvtColor((prob_map * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    cv2.imshow("image, segment contour, similarity map, selected point marked blue", np.hstack([img,im,prob_map]));
    cv2.waitKey()
