import pickle
import hashlib
import os
import argparse
import numpy as np
import torch
import cv2
import Desnet


#################################
def text_to_rgb(text):
    # Use the MD5 hash function to get a hash object
    m = hashlib.md5()
    m.update(text.encode('utf-8'))

    # Get the digest as a hexadecimal string
    hash_hex = m.hexdigest()

    # Take the first 6 digits, to make an RGB color
    # Each color component is represented by 2 digits in the hash
    r = int(hash_hex[0:2], 16)
    g = int(hash_hex[2:4], 16)
    b = int(hash_hex[4:6], 16)
    return [r,g,b]
#########################################################################################3

# Read MatSeg Benchmark test set

##########################################################################################
class TestSetReader():
    def __init__(self,images_dir,data_dir,max_size= 900):

        # Constants
        self.max_size= max_size # if image is larger then this resizr
        self.image_dir = images_dir+"//"
        self.data_dir = data_dir + "//"
        # if not os.path.exists(self.out_dir): os.mkdir(self.out_dir+"//")
        self.img_list=[]
        for fl in os.listdir(self.image_dir):
         #   path = self.image_dir +"/"+fl
            if os.path.isfile(self.image_dir+fl):
                if  os.path.splitext(fl)[1][1:].lower() in  ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff']:
                                         self.img_list.append(fl)
        self.itr=0
###########################################################################################################################################3

# Draw annotated points on image

########################################################################################
    def draw_all_points_im(self,im,matdt):
        """Draw all selected points."""
        img=  im.copy()
        for ky in matdt:
            color = text_to_rgb(ky)
            for point in self.matdt[ky]["pt_im"]:
                x, y,s = point
                img[y-s:y + s, x-s:x + s] = color
        cv2.imshow("",   cv2.resize(img.copy(),[int(img.shape[1]),int(img.shape[0])]))
        cv2.waitKey(1000)

#############################################################################################################################333

# Extract material similarity  relations from text (read txt string with smilarity as text and turn into table)

#############################################################################################################
    def GetSimFromTxt(self,txt):


        lines=txt.split("\n")
        for ln in lines:

            if "(" in ln:
                weight = 0.3
            if "[" in ln:
                weight = 0.6
            if ("[" in ln and "(" in ln) or ("[" not in ln and "(" not in ln):
                print("EERRRRPRRPR more then one rule in same ")
                exit()
            ln=ln.replace("[","").replace("(","").replace(")","").replace("]","")



            for ky in ln.split(","):
                if ky not in self.matdt:
                    print("error non existant key relation")
                    exit()

                for ky2 in ln.split(","):
                    if ky2==ky: continue
                    if ky2 in self.matdt[ky]["sim"]:
                        print("error relations added twice",ky2,ky)
                    self.matdt[ky]["sim"][ky2]=weight


############################################################################################################################################3

# Get annotation data for a given image (points materials and materials relations)

##############################################################################################################
    def get_data(self,dr,read_relations=True):
        # with open(dr + "/data.json", 'r') as f:
        #     self.matdt = json.load(f)
        with open(dr + "/data.pkl", 'rb') as f:
            self.matdt = pickle.load(f) # get annotated point
        for ky in self.matdt.keys(): self.matdt[ky]["sim"] = {ky: 1}  # add self similarity
        if os.path.exists(dr + "group_relations.txt") and read_relations:
            relations_txt=open(dr + "group_relations.txt","r").read()
            self.GetSimFromTxt(relations_txt) # get material similarity relations
        return self.matdt


###############################################################################################################################################
    # Load MatSeg and Image and Annotation
    def load(self, step=1, only_read_with_soft_relations=False):

        while (True): # get next image

            if self.itr + step >= len(self.img_list) or self.itr + step < 0: return False
            self.itr += step
            if self.itr >= len(self.img_list):
                print("FINISHED")
                exit()
            subdir = self.data_dir + "/" + self.img_list[self.itr][:-4] + "/"


            if not only_read_with_soft_relations or os.path.exists(self.data_dir + "/" + self.img_list[self.itr][:-4] + "/group_relations.txt"):
                break

        # Scaling factors
       # img_pg = pygame.image.load(self.image_dir + self.img_list[self.itr])

        # Convert the Pygame surface to a numpy array

        img = cv2.imread(self.image_dir + self.img_list[self.itr])
        self.img_name = self.img_list[self.itr]
        print(self.image_dir + self.img_list[self.itr])

        matdt = self.get_data(subdir)
        #---Resize if needed-----------------------------------------------
        original_height,original_width,d =img.shape
        rat= np.max([original_height/self.max_size,original_width/self.max_size])
        if rat>1 and self.max_size>0:
            width = int(original_width/rat)
            height = int(original_height/rat)
            img = cv2.resize(img,(width,height))
            for ky in matdt:
                for i in range(matdt[ky]['pt_im'].__len__()):
                    matdt[ky]['pt_im'][i]=(matdt[ky]['pt_im'][i]/rat).round().astype(np.int16)
        else:
            width = original_width
            height = original_height
            #------------------------create mask for every point in every material (mask in which the point region is 1 and the rest is zero\)------------------------------------------------
        matmasks={}
        for ky in matdt:
            matmasks[ky] = {}
            im2=img.copy()
            for i in range(matdt[ky]['pt_im'].__len__()):
                x, y, radius = matdt[ky]['pt_im'][i]
                image = np.zeros_like(img[:,:,0])
                matmasks[ky][i+1]= cv2.circle(image, (x,y), radius, 1, thickness=-1)
            #     im2[:,:,0]*=(1-matmasks[ky][i+1])
            #     im2[:, :, 1] *= (1 - matmasks[ky][i + 1])
            #     cv2.imshow(str(i),im2)
            #     cv2.waitKey()
            # cv2.destroyAllWindows()
        return img,matdt ,width, height, matmasks