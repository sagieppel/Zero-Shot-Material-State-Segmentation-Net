#######################################################################33

# Evaluate net accuracy on the MatSeg Benchmark

#######################################################################3


import argparse
import numpy as np
import torch
import MatSegBenchmarkLoader as BM
import Desnet

parser = argparse.ArgumentParser(description='Evaluate On MatSeg Benchmark')
parser.add_argument('--model_path',default='logs/Defult.torch',type=str,help="to trained model")
parser.add_argument('--images_dir', default= "samples/MatSegBenchMark/images", type = str, help = 'MatSeg Benchmark image folder')
parser.add_argument('--data_dir', default = "samples/MatSegBenchMark/data", type = str, help =  'MatSeg Benchmark data folder')
parser.add_argument('--descriptor_length', default= 128, type=int, help='descriptor length in net embedding map output')
parser.add_argument('--max_im_size', default= 700, type=float, help='max image size in pixels (one dimension), if loaded image is larger shrink it')
parser.add_argument('--use_only_soft_similarity',default=False,type=bool, help='evaluate only cases (triplets) with partial similarites')
parser.add_argument('--use_only_hard_similarity',default=False,type=bool, help='evaluate only cases with hard similarites, ignore partial similarity between materials')
args = parser.parse_args()

if args.use_only_soft_similarity and args.use_only_hard_similarity:
    print("args.use_only_soft_similarity and args.use_only_hard_similarity  cant both be true ")
    exit()
#######################################################################################################################################
# Calculate Triplet Metrics Section D

def get_accuracy(fmap,matmasks):
    matdescs={}
    dsclist=[]
    matlist=[]
    for matky in matmasks:
        matdescs[matky]=[]
        for pntkey in matmasks[matky].keys():
            pnt=matmasks[matky][pntkey].copy()
            pnt=np.expand_dims(pnt,2)
            pnt=np.tile(pnt,[1,1,fmap.shape[2]])
            desc=(pnt*fmap).sum(0).sum(0)/pnt.sum()
            desc /= (desc ** 2).sum() ** 0.5
            matdescs[matky].append(desc)
            dsclist.append(desc)
            matlist.append(matky)

    dscmat = np.array(dsclist)

    accuracy_all=[]
    accuracy_mat={}
    for i in range(dscmat.shape[0]):
        match_list=(dscmat[i] * dscmat).sum(1) # similarity of point to all other points
        num_mat=(matlist[i]==np.array(matlist)).sum() - 1# num appearacnce of current material
        argsort = np.argsort(match_list)[::-1]
        sortmat = np.array(matlist)[argsort] # sort material type according to match level
        same_mat = (sortmat==matlist[i])
        cumulative_sum = np.cumsum(~same_mat) # sum of wrong match up to this point
      #  inv_cumulative_sum = np.cumsum(~same_mat[::-1])[::-1]
        # What the next part do: For each correct match (same_mat) count how many wrong match occur before it (cumulative_sum)
        # sum this up and divide by all correct matches. Thi give you the probability  that match between same materials points will have higher similarity then
        accuracy=((1-(cumulative_sum[1:]/cumulative_sum[-1]))*same_mat[1:]).sum()/same_mat[1:].sum()
        accuracy_all.append(accuracy)
        if not matlist[i] in accuracy_mat:
            accuracy_mat[matlist[i]]=[]
        accuracy_mat[matlist[i]].append(accuracy)
#------------------------------------------------------------------------------------------------------------------------
    accuracy_point = np.mean(accuracy_all)
    print("accuracy by point:",accuracy_point)
    avg_acc_by_mat=[]
    for ky in accuracy_mat.keys():
        avg_acc_by_mat.append(np.mean(accuracy_mat[ky]))
    accuracy_mat = np.mean(avg_acc_by_mat)
    print("accuracy by mat:",accuracy_mat)


    return accuracy_point,accuracy_mat

#######################################################################################################################################
def get_accuracy_side_by_side(fmapnp,fmap,matmasks):
    matdescs={}
    dsclist=[]
    matlist=[]
    for matky in matmasks:
        matdescs[matky]=[]
        for pntkey in matmasks[matky].keys():
            # pntnp=matmasks[matky][pntkey].copy()
            # pntnp=np.expand_dims(pntnp,2)
            # pntnp=np.tile(pntnp,[1,1,fmap.shape[2]])
            # descnp=(pntnp*fmapnp).sum(0).sum(0)/pntnp.sum()
            # descnp /= (descnp ** 2).sum() ** 0.5

            pnt = torch.tensor(matmasks[matky][pntkey]).cuda()
            pnt = pnt.unsqueeze(2)
            pnt = torch.tile(pnt, [1, 1, fmap.shape[2]])
            desc = (pnt * fmap).sum(0).sum(0)
            desc /= (desc ** 2).sum() ** 0.5
            desc = desc.cpu().numpy()
            matdescs[matky].append(desc)
            dsclist.append(desc)
            matlist.append(matky)

    dscmat = np.array(dsclist)

    accuracy_all=[]
    accuracy_mat={}
    for i in range(dscmat.shape[0]):
        match_list=(dscmat[i] * dscmat).sum(1) # similarity of point to all other points
        num_mat=(matlist[i]==np.array(matlist)).sum() - 1# num appearacnce of current material
        argsort = np.argsort(match_list)[::-1]
        sortmat = np.array(matlist)[argsort] # sort material type according to match level
        same_mat = (sortmat==matlist[i]) # List of correct incorrect match by order
        cumulative_sum = np.cumsum(~same_mat) # sum of wrong match up to this point
      #  inv_cumulative_sum = np.cumsum(~same_mat[::-1])[::-1]
        accuracy=((1-(cumulative_sum[1:]/cumulative_sum[-1]))*same_mat[1:]).sum()/same_mat[1:].sum() # Get accuracy for this point with all other point pairs
        accuracy_all.append(accuracy)
        if not matlist[i] in accuracy_mat:
            accuracy_mat[matlist[i]]=[]
        accuracy_mat[matlist[i]].append(accuracy)
#------------------------------------------------------------------------------------------------------------------------
    accuracy_point = np.mean(accuracy_all)
    print("current image:  accuracy by point triplet:",accuracy_point)
    avg_acc_by_mat=[]
    for ky in accuracy_mat.keys():
        avg_acc_by_mat.append(np.mean(accuracy_mat[ky]))
    accuracy_mat = np.mean(avg_acc_by_mat)
    print("current image:  accuracy by material triplet:",accuracy_mat)


    return accuracy_point,accuracy_mat
####################################################################################################################################

# Given list of points and feature mask find the probability that points on the same group are more similar
# compare to pair of points of different types


##################################################################################33
def get_accuracy_pytorch(fmap,matmasks,matdic,use_soft_similarity,use_only_soft_similarity,rand=False):
    # fmap=  predicted descriptro map
    # matdic = List of points belonging to different material according to GT
    # matmasks= masks of the annotated points according to gt

    matdescs={} # descriptor of all points arranged by materiasl
    dsclist=[] # Descriptors of all points in a list
    matlist=[] # materials of of all descriptors in dsclist
    # Get average descriptor per point by averaging  the descriptors in the point region
    fmap = torch.nn.functional.normalize(fmap, dim=2)
    for matky in matmasks:# go over all materials types
        matdescs[matky]=[]
        for pntkey in matmasks[matky].keys(): # go over all points in a given material and get their  average descriptor from the feature map (fmap)
            pnt=torch.tensor(matmasks[matky][pntkey]).cuda()
            pnt=pnt.unsqueeze(2)
            pnt = torch.tile(pnt, [1, 1, fmap.shape[2]]) # match point mask dimension to descriptor map dimension

            desc=(pnt*fmap).sum(0).sum(0) # get sum of descriptors inside the point area
            if rand==True:
                desc = torch.rand(desc.shape)
            desc /= (desc ** 2).sum() ** 0.5 # Normalize descriptor
            desc=desc.cpu().numpy()
            matdescs[matky].append(desc) # add descriptor to list
            dsclist.append(desc)
            matlist.append(matky)

    dscmat = np.array(dsclist)

    accuracy_all=[]
    accuracy_mat={}
    accuracy_first=[]
    accuracy_mat_first= {}
    # Check the probability that point match other point on same group vs point in different group
    # this is basically triplet loss given to points with different similarity to the anchor point (p1) can you predict which one is more similar
    # Also do top one is the most similar point to p1 is of same material as p1
    # points similarity is assign by cosine similarities of points descriptors
    for p1 in range(dscmat.shape[0]):
        match_list=(dscmat[p1] * dscmat).sum(1) # similarity of point to all other points
       #### num_mat=(matlist[p1]==np.array(matlist)).sum() - 1# num appearacnce of current material
        argsort = np.argsort(match_list)[::-1]
        sortmat = np.array(matlist)[argsort] # sort material type according to match level

        #******************************************
        if use_soft_similarity: # check accuracy assume sof similarity hence points in different groups (materials) can still have partial similarity
                relation2weight = matdic[matlist[p1]]['sim']
                weight_relations = np.zeros_like(sortmat,dtype=np.float32)# how similar are two points
                for ky in relation2weight:
                    weight_relations[sortmat == ky] = relation2weight[ky] # how similar are two points
                accuracy_rate_by_point=[]
                for p2 in range(1,len(weight_relations)): # go over a given pair of points and see how this pair ranked in GT and prediction compare to all other pairs with this specific point i

                    # False is all points which appear before (according to prediction) but have lower affinity accurding to GT
                    if (weight_relations[p2]==1 or weight_relations[p2]==0) and use_only_soft_similarity: continue
                    error_count = (weight_relations[p2]>weight_relations[:p2]).sum()
                    # False is all points which appear after (according to prediction) but have higher affinity accurding to GT
                    error_count += (weight_relations[p2] < weight_relations[p2:]).sum()
                    # Divide by total amount of possible false
                    total_possible_errors=(weight_relations[p2] < weight_relations[1:]).sum()+(weight_relations[p2] > weight_relations[1:]).sum()
                    if total_possible_errors == 0: continue
                    mean_error =  error_count/total_possible_errors
                    accuracy_rate_by_point.append(1-mean_error)
                if len(accuracy_rate_by_point)==0: continue

                accuracy = np.array(accuracy_rate_by_point).mean() # mean accuracy for all pairs involving the point
                top1 = weight_relations[1]>= weight_relations[1:].max() # check if first match is the same material (for top 1 statitics
        else: # assume hard similarity, point in same group are identical point in different groups (materials) have no similarity
                same_mat = (sortmat == matlist[p1])
                cumulative_sum = np.cumsum(~same_mat)  # sum of wrong match up to this point
                if same_mat[1:].sum()==0: continue
                # Match accuracy for this material
                accuracy=((1-(cumulative_sum[1:]/cumulative_sum[-1]))*same_mat[1:]).sum()/same_mat[1:].sum() # count how many errors you got up to places of same material
                top1 =    same_mat[1]

     #******************************************
        accuracy_first.append(top1) # probability that first match is of the same material
        accuracy_all.append(accuracy)
        if not matlist[p1] in accuracy_mat:
            accuracy_mat[matlist[p1]]=[]
            accuracy_mat_first[matlist[p1]]=[]
        accuracy_mat[matlist[p1]].append(accuracy)
        accuracy_mat_first[matlist[p1]].append(top1)
#---------------------------Mean accuracy for all points in the image---------------------------------------------------------------------------------------------
    accuracy_point = np.mean(accuracy_all) # triplet accuract
    accuracy_first  = np.mean(accuracy_first)# top 1 accuracy
  #  print("accuracy by point:",accuracy_point, " first match:",accuracy_first)
    avg_acc_by_mat=[] # accuracy by equal weight to each material (regardless of point number)
    avg_acc_by_mat_first = []
    for ky in accuracy_mat.keys():
        avg_acc_by_mat.append(np.mean(accuracy_mat[ky]))
        avg_acc_by_mat_first.append(np.mean(accuracy_mat_first[ky]))
    accuracy_mat = np.mean(avg_acc_by_mat)
    accuracy_mat_first = np.mean(avg_acc_by_mat_first)
   # print("accuracy by mat:",accuracy_mat, "accuracy by mat first:",accuracy_mat_first)


    return accuracy_point,accuracy_mat ,accuracy_first,accuracy_mat_first

#######################################################################################
if __name__ == "__main__":
    dnet = Desnet.Net(descriptor_depth=args.descriptor_length) # build net

    dnet.load_state_dict(torch.load(args.model_path)) # load net weights
    if torch.cuda.is_available(): dnet.cuda()
    dnet.eval()

    accuracy_point_list = [] # average triplet accuracy for weighted by point
    accuracy_mat_list = [] # average triplet accuracy weighted by material
    accuracy_point_first_list = []# average top1 accuracy for weighted by point
    accuracy_mat_first_list = [] # average top1 accuracy weighted by material
    reader = BM.TestSetReader(images_dir=args.images_dir, data_dir=args.data_dir,max_size=args.max_im_size) #900
    reader.itr = 0
   # for itr in range(80020):
    while (True):
        img,matLdt ,width, height, matmasks = reader.load(only_read_with_soft_relations=args.use_only_soft_similarity) # read next image and its annotation
        print("size",img.shape,"sum:",img.sum())
        # ---------Main traininf loop----------------------------------------------------------------------------------------------------------------------------
        print("------------------------------------",reader.itr,"------------------------")
        with torch.no_grad():
                img = np.expand_dims(img, 0)
                fmap = dnet.forward(img,img[:,:,:,0]*0, TrainMode=False) # predict descriptor map for image
            #****************************************************************************************************************888
             #   thresh,map_type=SegmentFromFeatureMap.single_segment_from_GT_points(fmap[0].swapaxes(0, 2).swapaxes(0, 1),img,matmasks,thresh,map_type)#******************************************
              #  SegmentFromFeatureMap.full_segmention_from_GT_points(fmap[0].swapaxes(0, 2).swapaxes(0, 1),img,matmasks)
           #********************************************************************************************************
                # Get statics for image prediction vs gt
                accuracy_point, accuracy_mat,accuracy_first,accuracy_mat_first = get_accuracy_pytorch(fmap[0].swapaxes(0, 2).swapaxes(0, 1),matmasks=matmasks,matdic=matLdt,use_soft_similarity=~args.use_only_hard_similarity,use_only_soft_similarity=args.use_only_soft_similarity,rand=False)
              #  accuracy_point, accuracy_mat, accuracy_first, accuracy_mat_first = get_accuracy_pytorch(torch.tensor(img[0]).float().cuda(), matmasks, rand=False)
                # accumalete statics
                accuracy_point_list.append(accuracy_point)
                accuracy_mat_list.append(accuracy_mat)
                accuracy_point_first_list.append(accuracy_first)
                accuracy_mat_first_list.append(accuracy_mat_first)
                # display statistics
                print(accuracy_mat_list.__len__()," accumlate statitics by material, Triplet:",np.mean(accuracy_mat_list))
                print(accuracy_point_list.__len__(), " accumlate statics by point, Triplet:", np.mean(accuracy_point_list))
                print(accuracy_mat_list.__len__(), " accumumulate  statics by material, Top 1:", np.mean(accuracy_mat_first_list))
                print(accuracy_point_list.__len__(), " accumumulate  statics by point, Top 1:", np.mean(accuracy_point_first_list))

#
#
#