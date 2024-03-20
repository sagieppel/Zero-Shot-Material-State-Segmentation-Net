# Descripor net, receive an image and predict  descriptor per pixel
import torchvision.models as models
import torch
import copy
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class Net(nn.Module):# Net for region based segment classification
######################Load main net (resnet 50) class############################################################################################################
        def __init__(self,descriptor_depth=128): # Load pretrained encoder and prepare net layers
            super(Net, self).__init__()
            self.skip_con_stitch_mode = "add" #Method for merging the skip connection layer to the scalling layer
            self.transpose_mode = False # Upscaling mode (transpose conv or simple resizing)
            self.features2keep = { # Skip connections layers

                 2: {"layer": 5, "downscale": 16, "upscale": 2}
                ,4: {"layer": 3, "downscale": 8, "upscale": 4}
                ,8: {"layer": 1, "downscale": 4, "upscale": 8}

            } # features to keep for the skip connection
            self.upscaling_layers = { # upsampling layers of the UNET upscaling

                 2: {"upscale": 2, "indepth": 1024, "outdepth": 512, "skip_depth": 512}
                ,4: {"upscale": 4, "indepth": 512, "outdepth": 256, "skip_depth": 256}
                ,8: {"upscale": 4, "indepth": 256, "outdepth": descriptor_depth, "skip_depth": 128}# 128, "skip_depth": 128}
            }

            self.build_encoder() # build enconder (basically Convnext)
            self.build_decoder() # PSP or aspp layer deeplapb/pyramid scale parsing
            self.build_upsample() # upsacaling layer of the feature map


################################################################################################################################################################
# encoder layer proccess the image
        def build_encoder(self):

# ---------------Load pretrained net----------------------------------------------------------
            self.Encoder1 = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)#resnet50(pretrained=True)

#--------------Replace First layer from 3 channel input (RGB) to 4 channel (RGB,ROI)
            old_bias = copy.deepcopy(self.Encoder1.features[0][0].bias.data)
            old_weight = copy.deepcopy(self.Encoder1.features[0][0].weight.data)
            self.Encoder1.features[0][0]= torch.nn.Conv2d(4, 128, kernel_size=(4, 4), stride=(4, 4)) # Add layers to masks and pointer point
            self.Encoder1.features[0][0].weight.data[:,:3,:,:] = old_weight
            self.Encoder1.features[0][0].weight.data[:, 3, :, :] = 0
            self.Encoder1.features[0][0].bias.data = old_bias
 ########################################################################################################################################
# middle layer Pyramid Scene Parsing PSP or deeplab ASPP
        def build_decoder(self):
            self.mode = "psp"
            if self.mode=="psp":
            # ---------------------------------PSP layer----------------------------------------------------------------------------------------
                self.PSPLayers = nn.ModuleList()  # [] # Layers for decoder
                self.PSPScales = [1, 1 / 2, 1 / 4, 1 / 8]  # scalesPyramid Scene Parsing PSP layer
                for Ps in self.PSPScales:
                    self.PSPLayers.append(nn.Sequential(
                        nn.Conv2d(1024, 512, stride=1, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(512),
                        ##nn.nn.LayerNorm((512), eps=1e-06),
                        nn.GELU()))
            # ----------------------------------------ASPP  deeplab layers (deep lab)-----------------------------------------------------------------------
            elif  self.mode=="aspp":
                self.ASPPLayers = nn.ModuleList()
                self.ASPPScales = [1, 4, 8, 16]  # scales ASPP deep lab 3 net
                for scale in self.ASPPScales:
                    self.ASPPLayers.append(nn.Sequential(
                        nn.Conv2d(1024, 512, stride=1, kernel_size=3, padding=(scale, scale), dilation=(scale, scale),bias=False),
                        nn.BatchNorm2d(512),
                        ##nn.LayerNorm((512), eps=1e-06),
                        nn.GELU()))
                        #, nn.BatchNorm2d(512), nn.GELU()))


            # -------------------------------------------------------------------------------------------------------------------
            self.SqueezeLayers = nn.Sequential(
                nn.Conv2d(2048, 1024, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.GELU())  # ,
                # nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(512),
                # nn.ReLU()
            # ------------------Skip conncetion layers for upsampling-----------------------------------------------------------------------------

 ###############################################################################################################################
# final upsampling block (use skip connections)

        def build_upsample(self):




            self.upsample_ModuleList = nn.ModuleList()


            for i in  self.upscaling_layers:
                dt= self.upscaling_layers[i]
                if self.transpose_mode:
                    layer = nn.Sequential(
                                      nn.ConvTranspose2d(dt["indepth"], dt["outdepth"], 2, stride=2),
                                      nn.BatchNorm2d(dt["outdepth"]),
                                      #nn.LayerNorm((512,), eps=1e-06),
                                      nn.GELU())
                    self.upscaling_layers[i]["upscale_layer"] =   layer
                    self.upsample_ModuleList.append(layer)

                                      # nn.BatchNorm2d(512),
                else:
                    layer =  nn.Sequential(
                    nn.Conv2d(dt["indepth"], dt["outdepth"], stride=1, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(dt["outdepth"]),
                    #nn.LayerNorm((dt["outdepth"],), eps=1e-06),# 1e-05
                    #nn.BatchNorm2d(512),
                    nn.GELU())
                    self.upscaling_layers[i]["upscale_layer"] = layer
                    self.upsample_ModuleList.append(layer)

                skiplayer = nn.Sequential(
                    nn.Conv2d(dt["skip_depth"], dt["outdepth"], stride=1, kernel_size=1, padding=0, bias=False),
                    #nn.BatchNorm2d(512),
                    nn.BatchNorm2d(dt["outdepth"]),
                  #  nn.LayerNorm((dt["outdepth"],), eps=1e-06),
                    nn.GELU())
                self.upscaling_layers[i]["skip_layer"] = skiplayer
                self.upsample_ModuleList.append(skiplayer)

            #    if self.skip_con_stitch_mode == "cat":


#######################################Pre process  input#############################################################################################3
# Preprocces input image and ROI mostly normalize RGB to 0-1 transform axis to pytorch
        def preproccess_input(self,Images, Mask,TrainMode=True):
            # ------------------------------- Convert from numpy to pytorch-------------------------------------------------------
            #if TrainMode:
            mode = torch.FloatTensor
            # else:
            #     mode = torch.half

            self.type(mode)
            b,h,w,d = Images.shape




            InpImages1 = torch.autograd.Variable(torch.from_numpy(Images), requires_grad=False).transpose(2,
                                                                                                          3).transpose(
                1, 2).type(mode)
            ROIMask1 = torch.autograd.Variable(torch.from_numpy(Mask.astype(np.float32)),
                                               requires_grad=False).unsqueeze(dim=1).type(mode)


            ###########################################################################
            # for ii in range(Mask.shape[0]):
            #     im=Images[ii].copy().astype(np.uint8)
            #     im[:,:,0][pointer_mask[ii]>0]=0
            #     im[y[ii]-5:y[ii]+5, x[ii]-5:x[ii]+5, 1] = 255
            #     im[:, :, 2][Mask[ii] > 0] = 0
            #     cv2.imshow(str(ii),np.hstack([im,Images[ii].astype(np.uint8)]))
            #     cv2.waitKey()

            #########################################################################


            InpImages1 = InpImages1.to(device)
            ROIMask1 = ROIMask1.to(device)
            self.to(device)

            # -------------------------Normalize image-------------------------------------------------------------------------------------------------------
            RGBMean = [123.68, 116.779, 103.939]
            RGBStd = [65, 65, 65]
            for i in range(len(RGBMean)):
                InpImages1[:, i, :, :] = (InpImages1[:, i, :, :] - RGBMean[i]) / RGBStd[
                    i]  # Normalize image by std and mean
            # ============================Run net layers===================================================================================================
            inp_concat = torch.cat([InpImages1, ROIMask1], 1)
            return inp_concat
###########################################Run encoder########################################################################################333
# run encoder layer by layer
        def forward_encoder(self,x):
           for i in range(len(self.Encoder1.features)):
               x = self.Encoder1.features[i](x)
               for f in self.features2keep:
                   if self.features2keep[f]["layer"] == i:
                       self.features2keep[f]["features"] = x # save skip connection layers
            #   print(i, ")", x.shape)
           return x
###########################################Run encoder########################################################################################333

        def forward_midlayer(self,x): # run decoder
            if self.mode == "psp":
                PSPSize = (x.shape[2], x.shape[3])  # Size of the original features map
                PSPFeatures = []  # Results of various of scaled procceessing
                for i, PSPLayer in enumerate(
                        self.PSPLayers):  # run PSP layers scale features map to various of sizes apply convolution and concat the results
                    NewSize = (np.array(PSPSize) * self.PSPScales[i]).astype(np.int32)
                    if NewSize[0] < 1: NewSize[0] = 1
                    if NewSize[1] < 1: NewSize[1] = 1

                    # print(str(i)+")"+str(NewSize))
                    y = nn.functional.interpolate(x, tuple(NewSize), mode='bilinear', align_corners=False)
                    # print(y.shape)
                    y = PSPLayer(y)
                    y = nn.functional.interpolate(y, PSPSize, mode='bilinear', align_corners=False)
                    PSPFeatures.append(y)
                x = torch.cat(PSPFeatures, dim=1)
                x = self.SqueezeLayers(x)
            elif self.mode == "aspp":

                # ---------------------------------ASPP Layers--------------------------------------------------------------------------------
                ASPPFeatures = []  # Results of various of scaled procceessing
                for ASPPLayer in self.ASPPLayers:
                    y = ASPPLayer(x)
                    ASPPFeatures.append(y)
                x = torch.cat(ASPPFeatures, dim=1)
                x = self.SqueezeLayers(x)
            return x
###############################################Upsampling forward#########################################################################################################
      # Run upsampling block layer by layer
        def forward_upsample(self,x):
            for ii in range(1, 12):
                if ii in self.upscaling_layers:
                    if "upscale_layer" in self.upscaling_layers[ii]:
                        x = self.upscaling_layers[ii]["upscale_layer"](x)
                        if (ii in self.features2keep) and ("skip_layer" in self.upscaling_layers[ii]):
                            y = self.upscaling_layers[ii]["skip_layer"](self.features2keep[ii]["features"])
                            if y.shape[2] != x.shape[2] or y.shape[3] != x.shape[3]:
                                print("inconsistant upsampling scale")
                                x = nn.functional.interpolate(x, size=(y.shape[2], y.shape[3]), mode='bilinear',align_corners=False)
                                # combine skip layer with resized layer
                                if self.skip_con_stitch_mode == "add":
                                    x += y
                                elif self.skip_con_stitch_mode == "cat":
                                    x = torch.cat((y, x), dim=1)
            return x

###############################################Run prediction inference using the net ###########################################################################################################
# Run the full net on the image and get descriptor map with descriptor per pixel
        def forward(self,Images, Mask,TrainMode=True):
                x = self.preproccess_input(Images, Mask,TrainMode=TrainMode)
                x = self.forward_encoder(x)
                x = self.forward_midlayer(x)
                fmap  = self.forward_upsample(x)
                if TrainMode == False:
                   fmap = nn.functional.interpolate(fmap, size=(Images.shape[1], Images.shape[2]), mode='bilinear', align_corners=False)
                return fmap
                #
                # return fmap_upsample

##################################Loss##################################################################################################
# Calculate loss according to:(see section 6 Net And Training: https://arxiv.org/pdf/2403.03309.pdf)
# Get the average predicted descriptor of each material (from predicted map using materials regions extracted from GT map)
# Match the average descriptor of each material to the descriptor of each pixel (cosine similarity) to get pixel/material predicted probability
# Find crossetropy loss between GT material per pixel and Predicted material per pixel
        def crossentropy_loss_loops(self,GTMasksNP, pr_map,GTNumMasks,ROIs,temp=0.2):
           GTMasks=torch.tensor(GTMasksNP).to(device)
        #**************************************************

           # Assuming x is your input tensor of size [n, c, w, h]
           GTMasks = F.interpolate(GTMasks, size=(pr_map.shape[2],pr_map.shape[3]), mode='nearest')# resize GT to descriptor map size (net prediction is lower resolution then image)

           #*****************************************************
           gt_one_hot = torch.zeros([pr_map.shape[0],int(GTNumMasks.sum()),pr_map.shape[2],pr_map.shape[3]]).to(device)# one hot enconding of GT
           valid_mats= torch.zeros([int(GTNumMasks.sum())]) # # sum  materials might be invalid (not enough m samples) this list them
          #### gt_seg_map = np.zeros([pr_map.shape[0], pr_map.shape[2], pr_map.shape[3]])
           # one hot encoding map for gt
           pr_desc = F.normalize(pr_map, dim=1)
           gt_Descs=[]
           for ib in range(GTMasks.shape[0]): # go over all images in batch
               gt=GTMasks[ib][:GTNumMasks[ib]] # gt mask
               fmap = pr_desc[ib] # descriptor map of p
               for im in range(GTNumMasks[ib]):

                   gtmsk=gt[im]
                   bin_mask = gtmsk>0.5 # get region (mask) of specific material (according to GT)
                   if bin_mask.sum()==0: # if material region is empty there is  a problem
                  #     continue#*************************************************************
                       print("ERRROR bin sum 0")
                       continue
                   #    raise("bin sum errpr")
                    #   x=sdfs
                   #if bin_mask.sum()==0: continue
                   #**********Get descriptor by averaging region ***********************************************************************************
                   gtmsk= torch.tile(bin_mask,[fmap.shape[0],1,1]) # expand bin_mask to match the dimension of the feature vector
                   # if mode==1:
                   #     desc = torch.sqrt((gtmsk * fmap * fmap).sum(1).sum(1) / bin_mask.sum()) # Get average descriptor of material not used
                   # else:
                   # Get the average predicted descriptor of each material (from predicted map using materials regions extracted from GT map)
                   desc = (gtmsk * fmap).sum(1).sum(1) / bin_mask.sum()
                   desc = F.normalize(desc, dim=0)
                       # if  not desc.sum()==0:
                       #     desc = F.normalize(desc, dim=0)
                       # else:
                       #     print("something wrong")

                   gt_Descs.append(desc) # descriptor per material
                   #***************Create GT one hot*******************************************
                   mat_indx = len(gt_Descs) - 1
                   gt_one_hot[ib,mat_indx]=gt[im] # basically the one hot in the  material indx is the material mask
                   valid_mats[mat_indx] = 1
           ############Get predicted probability masks by matching descriptors##############################################################
           # Match the average descriptor of each material to the descriptor of each pixel (cosine similarity) to get pixel/material predicted probability
           gt_seg_map = torch.argmax(gt_one_hot, 1) # find the segment map by finding the material with the higher probability for each pixel
           desc_matrix = torch.stack(gt_Descs) # Turn descriptors to single tensor now you  can find the match between the descriptor image and each of the materials
           desc_matrix = desc_matrix.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
           pr_desc = pr_desc.unsqueeze(1)
           prd_logit =(desc_matrix * pr_desc).sum(dim=2) # cosine similarity between the material descriptor and the pixel descriptot give similarity of each material to each pixel
           #(desc_matrix[2,:]*pr_desc[1,:,300,305]).sum()=prd_logit[1,2,300,305]
           prd_prob = F.softmax(prd_logit/temp, dim=1) # transform similarities to probabilities



           # Find crossetropy loss between GT material per pixel and Predicted material per pixel
           torch_roi = torch.tensor(ROIs).to(device)
           torch_roi = F.interpolate(torch_roi.unsqueeze(1),  size=(pr_map.shape[2],pr_map.shape[3]), mode='nearest').squeeze(1)  # Remove the channel d
           gt_one_hot = gt_one_hot[:, valid_mats > 0]
           loss = -((gt_one_hot*torch.log(prd_prob)).mean(1) * torch_roi).mean()
           prd_seg_map = np.argmax(prd_prob.cpu().detach().numpy(), 1)
           return loss,prd_prob,prd_seg_map, gt_seg_map,gt_one_hot, torch_roi



##################################Binary Loss##################################################################################################

       #  def binary_loss_loops(self,GTMasksNP, pr_map,GTNumMasks,ROIs,mode=2,temp=0.2):
       #     GTMasks=torch.tensor(GTMasksNP).to(device)
       #  #**************************************************
       #     import torch.nn.functional as F
       #
       #     # Assuming x is your input tensor of size [n, c, w, h]
       #     GTMasks = F.interpolate(GTMasks, size=(pr_map.shape[2],pr_map.shape[3]), mode='nearest')#, align_corners=False)
       #
       #     #*****************************************************
       #     gt_one_hot = torch.zeros([pr_map.shape[0],int(GTNumMasks.sum()),pr_map.shape[2],pr_map.shape[3]]).to(device)
       #     valid_mats= torch.zeros([int(GTNumMasks.sum())]) # # sum  materials might be invalid (not enough m samples) this list them
       #    #### gt_seg_map = np.zeros([pr_map.shape[0], pr_map.shape[2], pr_map.shape[3]])
       #     # one hot encoding map for gt
       #     pr_desc = F.normalize(pr_map, dim=1)
       #     gt_Descs=[]
       #     prd_seg_map = torch.zeros_like(pr_map[:, 0, :, :])
       #     #*******Get descriptor per material by average material region
       #     loss = 0
       #     for ib in range(107021,GTMasks.shape[0]):
       #         gt=GTMasks[ib][:GTNumMasks[ib]] # gt mask
       #         fmap = pr_desc[ib]
       #         fmap = F.normalize(fmap, dim=0)
       #         torch_roi = torch.tensor(ROIs).to(device)
       #         for im in range(GTNumMasks[ib]):
       #             gtmsk=gt[im]
       #             bin_mask = gtmsk>0.5
       #             if bin_mask.sum()==0:
       #            #     continue#*************************************************************
       #                 print("ERRROR bin sum 0")
       #                 continue
       #             #    raise("bin sum errpr")
       #              #   x=sdfs
       #             #if bin_mask.sum()==0: continue
       #             #**********Get descriptor by averaging region ***********************************************************************************
       #             gtmsk= torch.tile(bin_mask,[fmap.shape[0],1,1]) # expand bin_mask to match the dimension of the feature vector
       #             # if mode==1:
       #             #     desc = torch.sqrt((gtmsk * fmap * fmap).sum(1).sum(1) / bin_mask.sum())
       #             # else:
       #             desc = (gtmsk * fmap).sum(1).sum(1) / bin_mask.sum()
       #             desc = F.normalize(desc, dim=0)
       #             prob_map=(desc.unsqueeze(1).unsqueeze(2) * fmap).sum(0)
       #
       #             loss += -(bin_mask.float()*torch.log(torch.abs(prob_map)+0.000001)).mean()
       #             loss += -((1-bin_mask.float()) * torch.log(1-torch.abs(prob_map) + 0.000001)).mean()
       #             if torch.isnan(loss):
       #                 print("dfdf")
       #
       #                 # if  not desc.sum()==0:
       #                 #     desc = F.normalize(desc, dim=0)
       #                 # else:
       #                 #     print("something wrong")
       #
       #             gt_Descs.append(desc) # the material descriptor is the average of the desciptors within the material gt mask
       #             #***************Create GT onehot*******************************************
       #             mat_indx = len(gt_Descs) - 1
       #             gt_one_hot[ib,mat_indx]=gt[im] # basically the one hot in the  material indx is the material mask
       #             valid_mats[mat_indx] = 1
       #             prd_seg_map[ib][prob_map > 0.5] = mat_indx
       #          ####   gt_seg_map[ib][GTMasksNP[ib][im]>0.8]= mat_indx# update gt segmentation map
       #     ############Get predicted probability masks by matching descriptors##############################################################
       #     gt_seg_map = torch.argmax(gt_one_hot, 1) # find the segment map by finding the material with the higher probability for each pixel
       # #    desc_matrix = torch.stack(gt_Descs) # Turn descriptors to single tensor now you  can find the match between the descriptor image and each of the materials
       # #    desc_matrix = desc_matrix.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
       #   #  pr_desc = pr_desc.unsqueeze(1)
       #  #   prd_logit =(desc_matrix * pr_desc).sum(dim=2)
       #     #(desc_matrix[2,:]*pr_desc[1,:,300,305]).sum()=prd_logit[1,2,300,305]
       #   #  prd_prob = F.softmax(prd_logit/temp, dim=1) # now we have match probability and can use it as a simple semantic segmentation
       #
       #     # torch_roi = torch.tensor(ROIs).to(device)
       #     ##torch_roi = F.interpolate(torch_roi.unsqueeze(1),  size=(pr_map.shape[2],pr_map.shape[3]), mode='nearest').squeeze(1)  # Remove the channel d
       #     # gt_one_hot = gt_one_hot[:, valid_mats > 0]
       #     # loss = -((gt_one_hot*torch.log(prd_prob)).mean(1) * torch_roi).mean()
       #     # prd_seg_map = np.argmax(prd_prob.cpu().detach().numpy(), 1)
       #     gt_seg_map = F.interpolate(gt_seg_map.float().unsqueeze(1), size=(GTMasksNP.shape[2], GTMasksNP.shape[3]),mode='nearest').squeeze(1)
       #     prd_seg_map = F.interpolate(prd_seg_map.float().unsqueeze(1), size=(GTMasksNP.shape[2], GTMasksNP.shape[3]),mode='nearest').squeeze(1)
       #     return loss,prd_seg_map.cpu().numpy(), gt_seg_map,gt_one_hot, torch_roi
# if __name__ == "__main__":
#     net=Net()