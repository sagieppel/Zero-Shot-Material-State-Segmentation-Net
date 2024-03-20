# Reader for MatSegDataset
import numpy as np
import os
import random
import cv2
import json
import threading
import random


############################################################################################################
#########################################################################################################################
class Reader:
    # Initiate reader and define the main parameters for the data reader
    def __init__(self, TrainDir, MaxBatchSize=100, MinSize=500, MaxSize=2000, MaxPixels=800 * 800 * 5,TrainingMode=True, Suffle=False):

        self.MaxBatchSize = MaxBatchSize  # Max number of image in batch
        self.MinSize = MinSize  # Min image width and hight in pixels
        self.MaxSize = MaxSize  # Max image width and hight in pixels
        self.MaxPixels = MaxPixels  # Max number of pixel in all the batch (reduce to solve oom out of memory issues)
        self.Epoch = 0  # Training Epoch
        self.itr = 0  # Training iteratation
        # ----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
        self.annlist = []
        print("Creating file list for reader this might take a while")

        for uu, dname in enumerate(os.listdir(TrainDir)):
            ###if uu > 100: continue
            print(uu, dname)
            s = {}
            if os.path.exists(TrainDir + "/" + dname + "/Finished.txt"): # only folder with finish.txt file considered valid
                s["dir"] = TrainDir + "/" + dname + "/"
                s["masks"] = []
                for fl in os.listdir(s["dir"]):
                    if fl == "RGB__RGB.jpg":  s["ImageFile"] = s["dir"] + "/" + fl # image
                    if fl[:4] == "mask":  s["masks"].append(s["dir"] + "/" + fl) # segmentation maps
                    if fl == "ObjectMaskOcluded.png":  s["ROI"] = s["dir"] + "/" + fl # ROI

                if len(s.keys()) == 4 and len(s['masks']) > 0:
                    self.annlist.append(s)

        if Suffle:
            np.random.shuffle(self.annlist)
        #
        print("All cats "+str(len(self.annlist)))
        if TrainingMode: self.StartLoadBatch()

    #############################################################################################################################
    # Crop and resize image and mask and Object mask to feet batch size
    def CropResize(self, Img, MatMasks, ROImask, Hb, Wb):
        # ========================resize image if it too small to the batch size==================================================================================
        bbox = cv2.boundingRect(ROImask.astype(np.uint8))
        [h, w, d] = Img.shape
        Rs = np.max((Hb / h, Wb / w))
        Wbox = int(np.floor(bbox[2]))  # ROI Bounding box width
        Hbox = int(np.floor(bbox[3]))  # ROI Bounding box height
        if Wbox == 0: Wbox += 1
        if Hbox == 0: Hbox += 1

        Bs = np.min((Hb / Hbox, Wb / Wbox))
        if Rs > 1 or (Bs < 1 and np.random.rand() < 0.3):# or Bs < 1 or np.random.rand() < 0.3:  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
            h = int(np.max((h * Rs, Hb)))
            w = int(np.max((w * Rs, Wb)))
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            for i in range(len(MatMasks)):
                MatMasks[i] = cv2.resize(MatMasks[i].astype(float), dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            ROImask = cv2.resize(ROImask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            bbox = (np.float32(bbox) * Rs.astype(np.float32)).astype(np.int32)

        # =======================Crop image to fit batch size===================================================================================
        x1 = int(np.floor(bbox[0]))  # Bounding box x position
        Wbox = int(np.floor(bbox[2]))  # Bounding box width
        y1 = int(np.floor(bbox[1]))  # Bounding box y position
        Hbox = int(np.floor(bbox[3]))  # Bounding box height

        if Wb > Wbox:
            Xmax = np.min((w - Wb, x1))
            Xmin = np.max((0, x1 - (Wb - Wbox) - 1))
        else:
            Xmin = x1
            Xmax = np.min((w - Wb, x1 + (Wbox - Wb) + 1))

        if Hb > Hbox:
            Ymax = np.min((h - Hb, y1))
            Ymin = np.max((0, y1 - (Hb - Hbox) - 1))
        else:
            Ymin = y1
            Ymax = np.min((h - Hb, y1 + (Hbox - Hb) + 1))

        if Ymax <= Ymin:
            y0 = Ymin
        else:
            y0 = np.random.randint(low=Ymin, high=Ymax + 1)

        if Xmax <= Xmin:
            x0 = Xmin
        else:
            x0 = np.random.randint(low=Xmin, high=Xmax + 1)

        # Img[:,:,1]*=PartMask
        # misc.imshow(Img)

        Img = Img[y0:y0 + Hb, x0:x0 + Wb, :]
        for i in range(len(MatMasks)):
            MatMasks[i] = MatMasks[i][y0:y0 + Hb, x0:x0 + Wb]
        ROImask = ROImask[y0:y0 + Hb, x0:x0 + Wb]
        # ------------------------------------------Verify shape match the batch shape----------------------------------------------------------------------------------------
        if not (Img.shape[0] == Hb and Img.shape[1] == Wb):
            Img = cv2.resize(Img, dsize=(Wb, Hb), interpolation=cv2.INTER_LINEAR)
            for i in range(len(MatMasks)):
                MatMasks[i] = cv2.resize(MatMasks[i][y0:y0 + Hb, x0:x0 + Wb, :], interpolation=cv2.INTER_LINEAR)
            ROImask = cv2.resize(ROImask, dsize=(Wb, Hb), interpolation=cv2.INTER_LINEAR)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return Img, MatMasks, ROImask
        # misc.imshow(Img)

    #################################################Generate Annotaton mask#############################################################################################################333


    ######################################################Augmented mask##################################################################################################################################
    # Augment image
    def Augment(self, Img, MatMasks, ROIMask, prob=1):
        Img = Img.astype(np.float32)
        if np.random.rand() < 0.5:  # flip left right
            Img = np.fliplr(Img)
            ROIMask = np.fliplr(ROIMask)
            for i in range(len(MatMasks)):
                MatMasks[i] = np.fliplr(MatMasks[i])

        if np.random.rand() < 0.5:  # flip up down
            Img = np.flipud(Img)
            ROIMask = np.flipud(ROIMask)
            for i in range(len(MatMasks)):
                MatMasks[i] = np.flipud(MatMasks[i])
        #
        # if np.random.rand() < prob: # resize
        #     r=r2=(0.6 + np.random.rand() * 0.8)
        #     if np.random.rand() < prob*0.2:  #Strech
        #         r2=(0.65 + np.random.rand() * 0.7)
        #     h = int(PartMask.shape[0] * r)
        #     w = int(PartMask.shape[1] * r2)
        #     Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        #     PartMask = cv2.resize(PartMask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        #     AnnMap = cv2.resize(AnnMap.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        if np.random.rand() < 0.035:  # Add noise
            noise = np.random.rand(Img.shape[0], Img.shape[1], Img.shape[2]) * 0.2 + np.ones(Img.shape) * 0.9

            Img *= noise
            Img[Img > 255] = 255
            #
        if np.random.rand() < 0.2:  # Gaussian blur
            Img = cv2.GaussianBlur(Img, (5, 5), 0)

        if np.random.rand() < 0.25:  # Dark light
            Img = Img * (0.5 + np.random.rand() * 0.65)
            Img[Img > 255] = 255
        # if np.random.rand() < prob:  # Dark light
        #     Img = Img * (0.5 + np.random.rand() * 0.7)
        #     Img[Img>255]=255

        if np.random.rand() < 0.2:  # GreyScale
            Gr = Img.mean(axis=2)
            r = np.random.rand()

            Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
            Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
            Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)

        return Img, MatMasks, ROIMask

    #######################################################################################################################################################
    # Display readed batch on screen
    def Display(self, img, MatMask, ROI,txt=""):
        img_cat = img.copy()
        tmp = img.copy()
        tmp[:, :, 0][ROI < 0.2] = 255
        tmp[:, :, 1][ROI < 0.2] = 0
        img_cat = np.concatenate([img_cat, tmp], 1)
        if type(MatMask) == list:
            for msk in MatMask:
                if msk.sum() == 0: continue
                tmp = img.copy()
                tmp[:, :, 0][msk > 0.2] = 255
                tmp[:, :, 1][msk > 0.2] = 0
                img_cat = np.concatenate([img_cat, tmp], 1)
        else:
            for i in range(MatMask.shape[0]):
                tmp = img.copy()
                msk = MatMask[i]
                if msk.sum() == 0: continue
                tmp[:, :, 0][msk > 0.2] = 255
                tmp[:, :, 1][msk > 0.2] = 0
                img_cat = np.concatenate([img_cat, tmp], 1)
        h, w, d = img_cat.shape
        r = 1500 / w
        h = int(h * r)
        w = int(w * r)
        cv2.destroyAllWindows()
        cv2.imshow(txt, cv2.resize(img_cat, (w, h)).astype(np.uint8))
        cv2.waitKey()

    #######################################################################################################################################################\

    # Normalized

    #########################################################################################################################################################
    # Normalize per pixel probability map per material by combining all the GT materials in every pixel and finiding their relative weights
    def Normalize(self, MatMask, ROImask):
        ROImask = (ROImask>200).astype(np.float32)
        mtnp=np.stack(MatMask) # Stack all materials maps
        mtsm = mtnp.sum(0)
        mtsm[mtsm<10]=0.1 # ignore lower value they tend to be noise
        normMask = mtnp/mtsm# find the relative weight of all materials in given pixel
        normMask[normMask>1.001]=0
        one_hot_norm=((normMask == normMask.max(0)) * (normMask > 0)).astype(np.float32)
        ROImask=(mtsm>0.1).astype(np.float32)
        MatMask=list(normMask)
        # mtnp
        # sum_mask = np.zeros_like(MatMask[0],dtype=np.float32)
        # for i, msk in enumerate(MatMask):
        #     MatMask[i] = MatMask[i].astype(np.float32) / 255
        return MatMask, ROImask

    ########################################################################################################################################################
    # load image for the batch and its annotation
    # ==========================Read image annotation and data===============================================================================================
    def LoadNext(self, batch_pos, Hb=-1, Wb=-1):
        while(True): # choose random image priorotizee images with more materials
           Nim = np.random.randint(len(self.annlist))
           if np.random.rand()>2/len(self.annlist[Nim]): break # make images with more materials more likely to be chosen

        Ann = self.annlist[Nim]
        # CatSize=100000000
        # --------------Read image--------------------------------------------------------------------------------
        Img = cv2.imread(Ann["ImageFile"])  # Load Image
        Img = Img[..., :: -1]# BGR->RGB
        if (Img.ndim == 2):  # If grayscale turn to rgb
            Img = np.expand_dims(Img, 3)
            Img = np.concatenate([Img, Img, Img], axis=2)
        Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
        # -------------------------Read annotation masks--------------------------------------------------------------------------------
        MatMasks = []
        for msk_path in Ann['masks']:
            mastmsk = cv2.imread(msk_path, 0).astype(float)
            MatMasks.append(mastmsk.astype(float))
        ROIMask = cv2.imread(Ann["ROI"], 0)  # Load mask

        MatMasks, ROIMask = self.Normalize(MatMasks, ROIMask) # combine seperate materials annotation masks into single probability mask

        # -----------------------------------Crop and resize-----------------------------------------------------------------------------------------------------
        #   self.Display(img=Img, MatMask=MatMasks, ROI=ROIMask,txt="raw")
        if not Hb == -1:
            Img, MatMasks, ROIMask = self.CropResize(Img, MatMasks, ROIMask, Hb, Wb)
        # -------------------------Augment-----------------------------------------------------------------------------------------------
        #  self.Display(img=Img, MatMask=MatMasks, ROI=ROIMask, txt="cropped")
        if np.random.rand() < 0.5:
            Img, MatMasks, ROIMask = self.Augment(Img, MatMasks, ROIMask)
        # self.Display(img=Img, MatMask=MatMasks, ROI=ROIMask, txt="augmented")
        # -------------------------Turn Masks list into numpy matrix---------------------------------------------------------------------------------------------------------
        # Find the non-zero matrices
        non_zero_masks = [matrix for matrix in MatMasks if not np.all(matrix <= 0.5)]
        for msk1 in non_zero_masks:
            if (msk1>0.5).sum()==0:
                  print("Whhhaaat?")
                  raise("what")
                  x=dfdf

        if len(non_zero_masks) == 0:
            return self.LoadNext(batch_pos, Hb, Wb)
        # Stack the non-zero matrices into a 3D array
        MatMasks = np.stack(non_zero_masks, axis=0)  # stack along last axis

        #self.Display(img=Img, MatMask=MatMasks, ROI=ROIMask, txt="stacked")

        # ---------------------------------------------------------------------------------------------------------------------------------
        self.BROIMask[batch_pos] = ROIMask
        self.BImgs[batch_pos] = Img
        self.BMasks[batch_pos][:MatMasks.shape[0]] = MatMasks[:self.BMasks[batch_pos].shape[0]]
        self.BNumMasks[batch_pos] = MatMasks.shape[0]

    ############################################################################################################################################################
    # Start load batch of images, segment masks, ROI masks, and pointer points for training MultiThreading s
    def StartLoadBatch(self, max_materials_per_img=10):
        # =====================Initiate batch=============================================================================================
        while True: # choose random size for batch
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width
            Hb = int(Hb / 32) * 32 # number of downsamples in convnext
            Wb = int(Wb / 32) * 32
            BatchSize = np.int32(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))
            if Hb * Wb < self.MaxPixels and BatchSize>1: break

      ##  BatchSize=1 #**************************
        self.BImgs = np.zeros((BatchSize, Hb, Wb, 3))  # List of all images
        self.BROIMask = np.zeros((BatchSize, Hb, Wb)) # List of all ROI Maps
      #  self.BPointMasks = np.zeros((BatchSize, 5, Hb, Wb))
        self.BMasks = np.zeros((BatchSize, max_materials_per_img, Hb, Wb)) # List of all material masks
       # self.BPointsXY = np.zeros((BatchSize, 2), dtype=np.uint32)
       ## self.BNumPointMasks = np.zeros((BatchSize), dtype=np.uint32)
        self.BNumMasks = np.zeros((BatchSize), dtype=np.uint32) # Number of material masks
        # ===============Select images for next batch
        # ====================Start reading data multithreaded===========================================================
        self.thread_list = []
        for pos in range(BatchSize):
            th = threading.Thread(target=self.LoadNext, name="thread" + str(pos), args=(pos, Hb, Wb))
            self.thread_list.append(th)
            th.start()
        self.itr += BatchSize

    ##################################################################################################################
    def SuffleFileList(self):
        random.shuffle(self.FileList)
        self.itr = 0

    ###########################################################################################################
    # Wait until the data batch loading started at StartLoadBatch is finished (for syncing multi threads reader)
    def WaitLoadBatch(self):
        for th in self.thread_list:
            th.join()

    ########################################################################################################################################################################################
    def LoadBatch(self):
        # Load batch for training (muti threaded  run in parallel with the training proccess)
        # For training
        self.WaitLoadBatch()

        Imgs = self.BImgs
        ROIMask = self.BROIMask
       # PointMasks = self.BPointMasks
        Masks = self.BMasks
        #PointsXY = self.BPointsXY
     #   NumPointMasks = self.BNumPointMasks
        NumMasks = self.BNumMasks
        self.StartLoadBatch()
        return Imgs.astype(np.float32), ROIMask.astype(np.float32), Masks.astype(np.float32), NumMasks.astype(np.int16)




########################################

if __name__ == "__main__":
    print("G")
    read = Reader(TrainDir=r"/media/breakeroftime/2T/Data_zoo/OutFolderMaterial_Segmentation/", MaxBatchSize=100,
                  MinSize=250, MaxSize=1000, MaxPixels=800 * 800 * 20, TrainingMode=True)
    Imgs, ROIMask, PointMasks, AllMasks, PointsXY, NumPointMasks, NumAllMasks = read.LoadBatch()

    for i in range(Imgs.shape[0]):
        read.Display(img=Imgs[i], ROI=ROIMask[i], MatMask=PointMasks[i], txt=str(i) + " pointers only",x=PointsXY[i][0], y=PointsXY[i][1])




