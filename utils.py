
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import matplotlib.patches as patches
from scipy import io as sio
from scipy.ndimage import gaussian_filter as gaussfilt

def GenerateHaarFeatureMasks(nbrHaarFeatures):
    """ GenerateHaarFeatureMasks
    Generate a random set of filter masks for Haar feature extraction.
    The masks will be of random sizes (within certain limits), at random
    locations, and of random direction (aligned along x or y).
    Both first order ([-1 1]-type) and second order features
    ([-1 2 -1]-type) are generated.

    The default size of 24x24 is assumed (can easily be changed below).

    Input:
        nbrHaarFeatures - Number of Haar feature masks to generate

    Output:
        haarFeatureMasks - A [24 x 24 x nbrHaarFeatures] matrix with
                           the Haar filter masks

    Written by Ola Friman, 2012
    """

    # We assume that the image size is 24x24 to reduce the
    # number of input parameters.
    imgSizex = 24    # Equal to number of columns
    imgSizey = 24    # Equal to number of rows

    # Intialize mask matrix
    haarFeatureMasks = np.zeros((imgSizey,imgSizex,nbrHaarFeatures))

    # Create feature masks one at the time
    for k in range(nbrHaarFeatures):

        # Randomize 1:st or 2:nd derivative type of Haar feature
        featureType = np.random.choice(2)

        # Randomize direction (x or y) of the filter
        featureDirection = np.random.choice(2)

        if featureType == 0:   # 1:st derivative type
            # Size of one field of the feature. For the 1:st deriviative
            # type, there are 2 fields, i.e., the actual size is twice as big
            xSize = 2 + np.random.choice(8)
            ySize = 2 + np.random.choice(8)  # Size between 2x(2 and 9)

            #
            # Find random origin so that the feature fits within the image
            xOriginMax = imgSizex - 2*xSize
            yOriginMax = imgSizey - 2*ySize
            xOrigin = np.random.choice(xOriginMax) # TODO:This might not need to start at 1, but 0
            yOrigin = np.random.choice(yOriginMax)

            # Generate feature
            if featureDirection == 0:   # x-direction
                haarFeatureMasks[yOrigin:yOrigin+2*ySize, xOrigin:      xOrigin+  xSize, k] = -1
                haarFeatureMasks[yOrigin:yOrigin+2*ySize, xOrigin+xSize:xOrigin+2*xSize, k] =  1
            else:                       # y-direction
                haarFeatureMasks[yOrigin:yOrigin+ySize         ,xOrigin:xOrigin+2*xSize, k] = -1
                haarFeatureMasks[yOrigin+ySize:yOrigin+2*ySize ,xOrigin:xOrigin+2*xSize, k] =  1

        elif featureType == 1:   # 2:nd derivative type
            # Size of one field of the feature. For the 1:st deriviative
            # type, there are 2 fields, i.e., the actual size is twice as big
            xSize = 2 + np.random.choice(5)
            ySize = 2 + np.random.choice(5)   # Size between 3x(2 and 6)

            # Find random origin so that the feature fits within the image
            xOriginMax = imgSizex - 3*xSize
            yOriginMax = imgSizey - 3*ySize
            xOrigin = np.random.choice(xOriginMax)
            yOrigin = np.random.choice(yOriginMax)

            # Generate feature
            if featureDirection == 0:   # x-direction
                haarFeatureMasks[yOrigin:yOrigin+3*ySize, xOrigin        :xOrigin+  xSize, k] = -1
                haarFeatureMasks[yOrigin:yOrigin+3*ySize, xOrigin+  xSize:xOrigin+2*xSize, k] =  2
                haarFeatureMasks[yOrigin:yOrigin+3*ySize, xOrigin+2*xSize:xOrigin+3*xSize, k] = -1
            else:                       # y-direction
                haarFeatureMasks[yOrigin:yOrigin+ySize           ,xOrigin:xOrigin+3*xSize, k] = -1
                haarFeatureMasks[yOrigin+ySize:yOrigin+2*ySize   ,xOrigin:xOrigin+3*xSize, k] =  2
                haarFeatureMasks[yOrigin+2*ySize:yOrigin+3*ySize ,xOrigin:xOrigin+3*xSize, k] = -1

    return haarFeatureMasks



def ExtractHaarFeatures(images,haarFeatureMasks):
    """ ExtractHaarFeatures
    Applies a number of Haar features from a stack of images.
    Input:
        images - A stack of images saved in a 3D matrix, first
                 image in image(:,:,1), second in image(:,:,2) etc.

        haarFeatureMasks - A stack of Haar feature filter masks saved in a 3D
                           matrix in the same way as the images. The haarFeatureMasks matrix is
                           typically obtained using the GenerateHaarFeatureMasks()-function

    Output:
        x - A feature matrix of size [nbrHaarFeatures,nbrOfImages] in which
            column k contains the result obtained when applying each Haar feature
            filter to image k.

    Written by Ola Friman, 2012
    """

    # Check that images and Haar filters have the same size
    if not images.shape[:2] == haarFeatureMasks.shape[:2]:
        raise Exception('Input image sizes do not match!')

    # Get number of images and number of features to extract
    nbrHaarFeatures = haarFeatureMasks.shape[2]
    nbrTrainingExamples = images.shape[2]

    # # Initialize matrix with feature values
    # x = np.zeros((nbrHaarFeatures,nbrTrainingExamples))

    # Extract features (using some Matlab magic to avoid one for-loop)
    x = haarFeatureMasks.reshape(-1,nbrHaarFeatures).T @ images.reshape(-1,nbrTrainingExamples)

    # NOTE, the above does the same as
    #for k = 1:nbrHaarFeatures
    #    for j = 1:nbrTrainingExamples
    #        x(k,j) = sum(sum(images(:,:,j).*haarFeatureMasks(:,:,k)));
    #    end
    #end

    return x


def GenerateTrainTestData(faces, nonfaces, haarFeatureMasks, nbrTrainImages=100, shuffle=False):
    # Shuffle images
    if shuffle:
        faces = faces[:,:,np.random.permutation(faces.shape[-1])]
        nonfaces = nonfaces[:,:,np.random.permutation(nonfaces.shape[-1])]
    
    # Create a training data set with examples from both classes.
    # Non-faces = class label y=-1, faces = class label y=1
    trainImages = np.concatenate((faces[:,:,:nbrTrainImages//2], nonfaces[:,:,:nbrTrainImages//2]), axis=2)
    xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks)
    yTrain = np.concatenate((np.ones(nbrTrainImages//2), -np.ones(nbrTrainImages//2)))

    # Create a test data set, using the rest of the faces and non-faces.
    testImages  = np.concatenate((faces[:,:,nbrTrainImages//2:], nonfaces[:,:,nbrTrainImages//2:]), axis=2)
    xTest = ExtractHaarFeatures(testImages,haarFeatureMasks)
    yTest = np.concatenate((np.ones(faces.shape[2]-nbrTrainImages//2), -np.ones(nonfaces.shape[2]-nbrTrainImages//2)))
    
    return trainImages, testImages, xTrain, yTrain, xTest, yTest


def PlotErrorGraphs(ErrTrain, ErrTest):
    ErrTestMinVal = np.min(ErrTest)
    ErrTestMinIdx = np.argmin(ErrTest)
    N = len(ErrTrain)
    
    plt.figure(figsize=(12,5))
    plt.plot(np.arange(1,N+1), ErrTrain, linewidth=2, label='Training error')
    plt.plot(np.arange(1,N+1), ErrTest , linewidth=2, label='Test error')
    plt.plot(ErrTestMinIdx+1, ErrTestMinVal, 'ok', markeredgewidth=2, markersize=10, markerfacecolor='None', label=f"Minimum test error ({ErrTestMinVal:.1f}% at {ErrTestMinIdx+1} classifiers)")
    plt.plot([-1,N+2], [7,7], '--k', linewidth=2, label='Target (7%)')
    plt.xlabel('Number of classifiers')
    plt.ylabel('Error rate [%]')
    plt.legend(loc='upper right')
    plt.ylim((0,np.max([ErrTrain,ErrTest])*1.05))
    plt.xlim((0,N+1))
    plt.grid()
    plt.savefig("ErrorPlot.png")
    plt.show()
    
    
def PlotClassifications(testImages, idxFaceCorr, idxFaceMis, idxNonFaceCorr, idxNonFaceMis, N=3, selectRandom=False):
    plt.figure(figsize=(4*N,4))
    
    if selectRandom:
        np.random.shuffle(idxFaceCorr)
        np.random.shuffle(idxFaceMis)
        np.random.shuffle(idxNonFaceCorr)
        np.random.shuffle(idxNonFaceMis)
    
    for i in range(N):
        ax = plt.subplot(2, 2*N, 1+i)
        plt.imshow(testImages[:,:,idxFaceCorr[i]], cmap='gray', vmin=0, vmax=256)
        ax.set_xticks([])
        ax.set_yticks([])
        if (i == 0):
            plt.ylabel("Correct")
    
    for i in range(N):
        ax = plt.subplot(2, 2*N, N+1+i)
        plt.imshow(testImages[:,:,idxNonFaceCorr[i]], cmap='gray', vmin=0, vmax=256)
        ax.set_xticks([])
        ax.set_yticks([])
    
    for i in range(N):
        ax = plt.subplot(2, 2*N, 2*N+1+i)
        plt.imshow(testImages[:,:,idxFaceMis[i]], cmap='gray', vmin=0, vmax=256)
        ax.set_xticks([])
        ax.set_yticks([])
        if (i == 0):
            plt.ylabel("Misclassified")

    for i in range(N):
        ax = plt.subplot(2, 2*N, 3*N+1+i)
        plt.imshow(testImages[:,:,idxNonFaceMis[-i]], cmap='gray', vmin=0, vmax=256)
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()
    

def PlotSelectedHaarFeatures(haarFeatureMasks, selectedIdx, N=None, shuffle=False):
    if N is None:
        N = len(selectedIdx)
        
    if shuffle:
        np.random.shuffle(selectedIdx)
        
    r = np.floor(np.sqrt(N)).astype('int')
    c = np.ceil(np.sqrt(N)+0.5).astype('int')

    plt.figure(figsize=(8,8*r/(c-0.5)))
    for i in range(N):
        plt.subplot(r,c,i+1)
        plt.imshow(haarFeatureMasks[:,:,selectedIdx[i]], cmap='RdBu_r', vmin=-2, vmax=2)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{selectedIdx[i]}")
    plt.suptitle('Chosen Haar features', fontweight="bold")
    plt.show()
    
    
def PlotHaarFeatureDemonstration(haar, face, nonface):
    plt.figure(figsize=(15,6))

    # Plot Haar-feature
    plt.subplot(2,4,(1,5))
    im = plt.imshow(haar, cmap='RdBu_r', vmin=-2, vmax=2)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([])
    plt.yticks([])
    plt.box("on")
    plt.title("Haar-feature")

    # Plot face
    plt.subplot(2,4,2)
    im = plt.imshow(face, cmap='gray', vmin=0, vmax=256)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([])
    plt.yticks([])
    plt.box("on")
    plt.title("Face")

    # Plot nonface
    plt.subplot(2,4,6)
    im = plt.imshow(nonface, cmap='gray', vmin=0, vmax=256)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([])
    plt.yticks([])
    plt.box("on")
    plt.title("Nonface")

    # Plot face
    plt.subplot(2,4,3)
    im = plt.imshow(face * haar, cmap='RdBu_r', vmin=-400, vmax=400)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([])
    plt.yticks([])
    plt.box("on")

    plt.subplot(2,4,7)
    im = plt.imshow(nonface * haar, cmap='RdBu_r', vmin=-400, vmax=400)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([])
    plt.yticks([])
    plt.box("on")

    ax = plt.subplot(2,4,4)
    ax.text(0.5, 0.5,
            r'$x = \sum \mathrm{Blue} + \sum \mathrm{Red} = %d$' % np.sum(face * haar),
           fontsize=14, horizontalalignment="center", verticalalignment="center")
    ax.axis("off")

    ax = plt.subplot(2,4,8)
    ax.text(0.5, 0.5,
            r'$x = \sum \mathrm{Blue} + \sum \mathrm{Red} = %d$' % np.sum(nonface * haar),
           fontsize=14, horizontalalignment="center", verticalalignment="center")
    ax.axis("off")

    plt.show()
    
    
def PlotSolvayHeatmap(cSolvay):
    Solvay = sio.loadmat('Data/solvay.mat')
    X = Solvay['X'].astype("float")
    img = Solvay['Image'].astype("float")
    coords = Solvay['Coords']
    
    hm = np.zeros_like(img)
    n  = np.zeros_like(img)
    for i,c in enumerate(cSolvay):
        hm[coords[i,0]:(coords[i,0]+24), coords[i,1]:(coords[i,1]+24)] += c
        n[coords[i,0]:(coords[i,0]+24), coords[i,1]:(coords[i,1]+24)] += 1
    hm /= n
    
    fig = plt.figure(figsize=(16,16))
    
    plt.subplot(2,1,1)
    plt.imshow(img, cmap="gray")
    plt.imshow(hm, cmap="jet", alpha=0.5, vmin=-1, vmax=np.percentile(hm,99.9))
    plt.colorbar(fraction=0.023, pad=0.01)
    plt.xticks([])
    plt.yticks([])
    plt.box("on")
    plt.title("Heatmap from AdaBoost classifier")
    
    plt.subplot(2,1,2)
    plt.imshow(img, cmap="gray")
    plt.colorbar(fraction=0.023, pad=0.01)
    plt.xticks([])
    plt.yticks([])
    plt.box("on")
    plt.title("Boundaries for the most likely regions")
    
    hm = gaussfilt(hm, 1.0)
    ax = plt.gca()
    p = np.percentile(hm[13:-13, 13:-13], 90)
    for idx,v in np.ndenumerate(hm):
        if (idx[0]<12) or (idx[1]<12) or (idx[0]>387) or (idx[1]>587):
            continue
        if (v > p) and (v == np.max(hm[idx[0]-12 : idx[0]+12, idx[1]-12 : idx[1]+12])):
            rect = patches.Rectangle((idx[1]-12, idx[0]-12), 24, 24, edgecolor="r", facecolor="none", linewidth=2.0)
            ax.add_patch(rect)
    
    fig.tight_layout()
    plt.savefig('Adaboost.png')
    plt.show()