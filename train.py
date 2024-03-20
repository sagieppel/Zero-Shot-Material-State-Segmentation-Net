# Train material state segmentation net on MatSeg

import os
import numpy as np
import torch
import argparse
import MatSegReader
#import Desnet_DeepLab as Desnet
import Desnet
import Statistics_Making
# input parameters
parser = argparse.ArgumentParser(description='Train Material Segmentation On MatSeg Dataset')
parser.add_argument('-data_dir',default="samples/train_data",type=str,help="path to dataset dir")
parser.add_argument('--MaxPixels', default= 700*700*2, type=int, help='max Size of input matrix in pixels H*W*BatchSize (reduce to solve cuda out of memory)')
parser.add_argument('--MaxImagesInBatch', default = 7, type=int, help='max images in a a batch (reduce to solve cuda out of memory)')
parser.add_argument('--temp', default= 0.2, type=float, help='temperature for softmax')
parser.add_argument('--minsize', default= 250, type=float, help='min image size in pixels (one dimension)')
parser.add_argument('--maxsize', default= 800, type=float, help='max image size in pixels (one dimension)')
parser.add_argument('--weight_decay', default= 4e-5, type=float, help='optimizer weight decay')
parser.add_argument('--learning_rate', default= 1e-5, type=float, help='optimizer learning rate')
parser.add_argument('--descriptor_depth', default= 128, type=int, help='depth of output descriptor')
parser.add_argument('--log_dir', default= r"logs/", type=str, help='log folder were train model will be saved')
parser.add_argument('--resume_training_from', default= r"", type=str, help='path to model to resume training from if "" ignore this ')
parser.add_argument('--auto_resume', default= True, type=bool, help='start training from existing last saved model (Defult.torch)')
parser.add_argument('--display', default= False, type=str, help='display readed data and results on screen for debug')
#parser.add_argument('--sam_weight_path', default= r"sam_vit_h_4b8939.pth", type=str, help='path to model to resume training from')

args = parser.parse_args()
if not os.path.exists(args.log_dir):os.mkdir(args.log_dir)
InitStep=0
if args.auto_resume:
    if os.path.exists(args.log_dir + "/Defult.torch"):
        args.resume_training_from=args.log_dir  + "/Defult.torch"
    if os.path.exists(args.log_dir +"/Learning_Rate.npy"):
        args.learning_rate=np.load(args.log_dir +"/Learning_Rate.npy")
    if os.path.exists(args.log_dir +"/itr.npy"): InitStep=int(np.load(args.log_dir +"/itr.npy"))




#*****************************************************************
#sam_checkpoint = "sam_vit_h_4b8939.pth"

device = "cuda"
model_type = "default"
import sys
sys.path.append("..")
# from segment_anything import sam_model_registry

dnet=Desnet.Net(descriptor_depth=args.descriptor_depth)
if args.resume_training_from!="": # Optional initiate full net
    dnet.load_state_dict(torch.load(args.resume_training_from))
optimizer = torch.optim.AdamW(params= dnet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# MatSeg Reader
reader = MatSegReader.Reader(TrainDir=args.data_dir, MaxBatchSize=args.MaxImagesInBatch,MinSize=args.minsize, MaxSize=args.maxsize, MaxPixels=args.MaxPixels, TrainingMode=True)
AVGLoss = {"IOUclass":0,"IOUpixel":0,"loss":0}

scaler = torch.cuda.amp.GradScaler()

#---------Main traininf loop----------------------------------------------------------------------------------------------------------------------------


for itr in range(InitStep,1000000):
    print("--------------",itr,"-----------------------------------------------------------")
    print("---------start reading----------------")
    Imgs, ROIMask,  GTMasks, GTNumMasks = reader.LoadBatch() # read batch
    print("Batch shape",ROIMask.shape, " sum masks",GTNumMasks.sum())
    if GTNumMasks.sum()==1: continue # ignore batch with only one material
    #************************Display data read*****************************************************************************

    print("---------Finished reading----------------")
    if args.display:
        for i in range(Imgs.shape[0]):
           reader.Display(img=Imgs[i], ROI=ROIMask[i], MatMask=GTMasks[i], txt="input Masks for image "+str(i))
    #************************************************************************************************************************
    #--------Decoder with training------------------------------------------------------------------------------

    with torch.cuda.amp.autocast():#############
            dnet.zero_grad()
            fmap = dnet.forward(Imgs, ROIMask*0,  TrainMode=True) # Predict descriptor/embedding from image

            # get contrastive loss (see section 6 Net And Training: https://arxiv.org/pdf/2403.03309.pdf)
            loss, prd_prob, prd_seg_map, gt_seg_map, gt_one_hot, ROIMask  = dnet.crossentropy_loss_loops(GTMasksNP=GTMasks, pr_map=fmap,ROIs=ROIMask,GTNumMasks=GTNumMasks,temp=args.temp)
            print("loss",loss)
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
   #--------Get statics and save models ()---------------------------------------------------------------------
            # Get statitics and save sample results in the log dir

            IOUclass,IOUpixel = Statistics_Making.get_statics(Imgs, prd_seg_map, gt_seg_map, gt_one_hot,logdir=args.log_dir,ROI=ROIMask,itr=itr,save=(itr%500)==0,display=args.display)

        #========update statitics running average=============================================================================
            fr = 1 / np.min([itr - InitStep + 1, 2000])
            AVGLoss["IOUclass"] = AVGLoss["IOUclass"]* (1 - fr) + fr * float(IOUclass)
            AVGLoss["IOUpixel"] = AVGLoss["IOUpixel"] * (1 - fr) + fr * float(IOUpixel)
            AVGLoss["loss"] = AVGLoss["loss"] * (1 - fr) + fr * float(loss)



            # ===================save statitics and displaye loss======================================================================================
            # --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
            if itr % 10 ==0:# Display accuracy (running average)
                txt=""
                for ky in AVGLoss:
                    txt += str(itr) + ky +" = "+str(AVGLoss[ky])
                print(txt)
            if itr%1000 ==0: # Dae accuracy (running average)
                with open(args.log_dir + "/"+str(itr)+".txt", "w") as file:
                    # Write the text to the file
                    file.write(txt)
            if itr % 2000 == 0 and itr > 0:  # Save model weight and other paramters in temp file once every 2000 steps
                print("Saving Model to file in " + args.log_dir + "/Defult.torch")
                torch.save(dnet.state_dict(), args.log_dir + "/Defult.torch")
                torch.save(dnet.state_dict(), args.log_dir + "/DefultBack.torch")
                print("model saved")
                np.save(args.log_dir + "/Learning_Rate.npy", args.learning_rate)
                np.save(args.log_dir + "/itr.npy", itr)
                torch.cuda.empty_cache()  # clean memory
            if itr % 20000 == 0 and itr > 0:  # Save model weight once every 20k steps permenant (with step number)
                print("Saving Model to file in " + args.log_dir + "/" + str(itr) + ".torch")
                torch.save(dnet.state_dict(), args.log_dir + "/" + str(itr) + ".torch")
                print("model saved")
