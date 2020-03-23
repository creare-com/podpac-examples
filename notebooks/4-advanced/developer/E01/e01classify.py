'''
ABOUT:
This python program will run an SVM classifier on an EO-1 ALI scene,
and output a GeoTiff containing the classified scene.

DEPENDS:
gdal
numpy
sklearn

AUTHORS:
Jake Bruggemann
Maria Patterson

HISTORY:
April 2014: Original script (beta).

USE:
For use on the Open Science Data Cloud public data commons.
> python classify.py YEAR DAY IMAGE OUTFILE.PNG
For example, classify an image of the Italian coast from 29/1/2014
> python classify.py 2014 029 EO1A1930292014029110PZ italyClassified.tif
'''
__author__ = 'Jake Bruggemann'
__version__ = 0.3

# Class: test
#
# Object to set up, run, and save results from an SVM classification
# 
import gdal,osr
import scipy.misc as mpimg
import numpy as np
import time
import csv
from sklearn import svm
import numpy.ma as ma
from sklearn.externals import joblib
import os.path
from PIL import Image
import matplotlib.pyplot as plt

class test(object):
    
    # initializer
    # 
    # str: filePre, prefix for EO-1 Scene files

    def __init__(self, filePre='', output='test.tif'):
        self.fullTrainSet = np.array([])
        self.fullTestSet = np.array([])
        self.trainSet = np.array([])
        self.trainLab = None
        self.filePre = filePre
        self.output = output
        self.createMetadata()
        self.bands = np.array([])
        self.testSet = np.array([])
        self.dims = None
        self.mask = None
        self.answers = None
        self.tester = None
        
    # addRatio
    #
    # Adds ratio of two bands to test / training set
    # 
    # (int1, int2): ratio, int1:int2 ratio to be added

    def addRatio(self,ratio):
        numerInd = self.bands[np.where(self.bands==ratio[0]-1)[0]]
        denomInd = self.bands[np.where(self.bands==ratio[1]-1)[0]]
        numerArr = np.reshape(self.fullTestSet[:,numerInd],(self.fullTestSet.shape[0],1))
        denomArr = np.reshape(self.fullTestSet[:,denomInd],(self.fullTestSet.shape[0],1))

        nTrain = np.reshape(self.fullTrainSet[:,numerInd],(self.fullTrainSet.shape[0],1))
        dTrain = np.reshape(self.fullTrainSet[:,denomInd],(self.fullTrainSet.shape[0],1))

        ratioArr = numerArr / denomArr
        trainArr = nTrain/dTrain

        if self.testSet.size ==0:
            self.testSet = ratioArr
            self.trainSet = trainArr
        else:
            self.testSet = np.concatenate((self.testSet,ratioArr),axis=1)
            self.trainSet = np.concatenate((self.trainSet,trainArr),axis=1)
        pass
        
    
    # addBand
    # 
    # Adds list of bands to test / training set
    # 
    # [int,..,ints]: bands, list of bands to be added
    
    def addBand(self,bands):
        for band in bands:
            test = np.reshape(self.fullTestSet[:,band],(self.fullTestSet.shape[0],1))
            train = np.reshape(self.fullTrainSet[:,band],(self.fullTrainSet.shape[0],1))
            if self.testSet.size== 0:
                self.testSet = test
                self.trainSet = train
            else:
                self.testSet = np.concatenate((self.testSet,test),axis=1)
                self.trainSet = np.concatenate((self.trainSet,train),axis=1)
                
        pass


    # setUpTest
    # 
    # Loads band GeoTiffs from gluster into test set

    def setUpTest(self):
        bandList = np.array(['_B02','_B03','_B04','_B05','_B06','_B07','_B08','_B09','_B10'])
        for band in np.arange(9):
            bandName = bandList[band]
            tifFile = gdal.Open(self.filePre+bandName+'_L1GST.TIF')
            rast = tifFile.GetRasterBand(1)
            arr = rast.ReadAsArray()
            if self.dims == None: self.dims = arr.shape
            arra = np.reshape(arr,(arr.size,1))

            if self.mask is None: self.mask = np.reshape((arra==0),arra.size)
            mArra = ma.masked_array(arra,self.mask)
            if self.fullTestSet.size == 0:
                self.fullTestSet = mArra                
                self.bands = np.array([band])
            else:
                self.fullTestSet = np.concatenate((self.fullTestSet,mArra),axis=1)            
                self.bands = np.concatenate((self.bands,np.array([band])))
        self.rescaleALI()
        self.aliSolarIrradiance()
        self.geometricCorrection()


    # setUpTrain
    #
    # Loads training set and bins Hyperion-based values to resemble ALI band coverage
    #
    # str: trainName, file name of training set

    def setUpTrain(self,trainName):
        trainSet = np.loadtxt(open(trainName),skiprows=1,delimiter=",")
        
        hypBandsNum = np.array([['009','010'],
                                ['011','012','013','014','015','016'], 
                                ['018','019','020','021','022','023','024','025'],
                                ['028','029','030','031','032','033'],
                                ['042','043','044','045'],
                                ['049','050','051','052','053','71','72','73','74'],
                                ['106','107','108','109','110','111','112','113','114','115'],
                                ['141','142','143','144','145','146','147','148','149','150',
                                 '151','152','153','154','155','156','157','158','159','160'],
                                ['193','194','195','196','197','198','199','200','201','202',
                                 '203','204','205','206','207','208','209','210','211','212',
                                 '213','214','215','216','217','218','219']])
        self.trainLabels = np.reshape(trainSet[:,-1],trainSet.shape[0])
        self.fullTrainSet = np.zeros([trainSet.shape[0],9])
        for i in np.arange(9):
            bandNums = hypBandsNum[i]
            for band in bandNums:
                self.fullTrainSet[:,i] = self.fullTrainSet[:,i]+trainSet[:,int(band)-1]
            self.fullTrainSet[:,i] = self.fullTrainSet[:,i]/len(bandNums)
        pass

    # createMetaData
    #
    # Loads metadata for ALI scene, and stores in test object for further reference

    def createMetadata(self):
        l1t = {}

        filename = self.filePre+"_MTL_L1GST.TXT"

        with open(filename) as l1tFile:
            last = l1t
            stack = []
            for line in l1tFile:
                if line.rstrip() == "END": break
                name, value = line.rstrip().lstrip().split(" = ")
                value = value.rstrip("\"").lstrip("\"")
                if name == "GROUP":
                    stack.append(last)
                    last = {}
                    l1t[value] = last
                elif name == "END_GROUP":
                    last = stack.pop()
                else:
                    last[name] = value

        self.metadata = l1t
        pass

    # rescaleALI
    #
    # Rescales ALI radiances in accordance with metadata 

    def rescaleALI(self):

        radianceScaling = self.metadata['RADIANCE_SCALING']
        bandScaling = np.zeros((1,self.bands.size))
        bandOffset = np.zeros((1,self.bands.size))
        
        for i in np.arange(self.bands.size):
            bandScaling[0,i] = float(radianceScaling['BAND' + str(self.bands[i]+2) + '_SCALING_FACTOR'])
            bandOffset[0,i] = float(radianceScaling['BAND' + str(self.bands[i]+2) + '_OFFSET'])

        self.fullTestSet =  (self.fullTestSet * bandScaling) + bandOffset   
        pass
    
    # geometricCorrection
    #
    # Corrects for geometric orientation of sun, according to metadata Part of converting
    # Radiance values to reflectance

    def geometricCorrection(self):
        earthSunDistance = np.array([[1,.9832], [15,.9836], [32,.9853], [46,.9878], [60,.9909],
                                     [74, .9945], [91, .9993], [106, 1.0033], [121, 1.0076], [135, 1.0109],
                                     [152, 1.0140], [166, 1.0158], [182, 1.0167], [196, 1.0165], [213, 1.0149],
                                     [227, 1.0128], [242, 1.0092], [258, 1.0057], [274, 1.0011], [288, .9972],
                                     [305, .9925], [319, .9892], [335, .9860], [349, .9843], [365, .9833],[366, .9832375]])
        
        julianDate = time.strptime(self.metadata["PRODUCT_METADATA"]["START_TIME"], "%Y %j %H:%M:%S").tm_yday
        eSD = np.interp( np.linspace(1,366,366), earthSunDistance[:,0], earthSunDistance[:,1] )
        esDist = eSD[julianDate-1]
        
        sunAngle = float(self.metadata["PRODUCT_PARAMETERS"]["SUN_ELEVATION"])
        sunAngle = sunAngle*np.pi/180.
        self.fullTestSet = np.pi * esDist**2 * self.fullTestSet / np.sin(sunAngle)
        pass
    
    # aliSolarIrradiance
    #
    # Corrects for solar flux for each ALI band, in accordance with metadata. Part of 
    # converting radiance values to reflectance
    
    def aliSolarIrradiance(self):
        Esun_ali = np.array([[2,1851.8], [3, 1967.6], [4, 1837.2], [5,1551.47], [6, 1164.53], [7,957.46], [8, 451.37], [9, 230.03], [10, 79.61]])
        self.fullTestSet = self.fullTestSet / np.reshape(Esun_ali[self.bands,1],(1,self.bands.size))
        pass

    # svmTrain
    #
    # Generates svm object and trains to the training data, has option to save SVM test
    # 
    # str: kern, Kernel used for SVM test, check sklearn documentation for more info
    # [float]: c, Optional Penalty parameter C for error term
    # [float]: gam, Optional Kernel coefficient
    # [str]: savePath, Optional Path to save SVM test to (called a pickle)

    def svmTrain(self,kern='rbf',c=100,gam=.01,savePath = None):
        clf = svm.SVC(kernel = kern,gamma=gam,C=c)    #set up svm
        print("Created Test...")
        clf.fit(self.trainSet,self.trainLabels)
        print("Fitted SVM")
        if savePath != None:
            joblib.dump(clf,savePath)
        else:
            self.tester = clf
        pass
    
    
    # svmTest
    #
    # Loads svm test pickle and classifies test set.
    # 
    # [str]: testFile, Optional filename of SVM pickle
    

    def svmTest(self,testFile=None):
        if testFile==None:
            clf = self.tester
        else:
            clf = joblib.load(testFile) #unpickle svmTest
        print("Loaded Test")
        self.answers = clf.predict(self.testSet)

        print("Predicted Answers")
        self.answers[self.mask] = 0
        self.answers = np.reshape(self.answers,[self.answers.size,1])
        return self.answers
    
    # writeTif
    #
    # Writes GeoTiff using values generated from svmTest function
    #
    # str: filename, File name of target GeoTiff.
        
    def writeTif(self, filepath=None):

        if filepath is None:
            filepath = self.output

        driver = gdal.GetDriverByName("GTiff")
        driver.Register()
        src = gdal.Open(self.filePre+"_B02_L1GST.TIF")
        
        cols = src.RasterXSize
        rows = src.RasterYSize
        srcBand = src.GetRasterBand(1)
        numBands = src.RasterCount
        datatype = srcBand.DataType
        dest = driver.Create(filepath,cols,rows,numBands,datatype)
        
        geoTransform = src.GetGeoTransform()
        dest.SetGeoTransform(geoTransform)
        
        proj = src.GetProjection()
        dest.SetProjection(proj)
        
        met = src.GetMetadata_List()
        dest.SetMetadata(met)
        
        band = dest.GetRasterBand(1)
        band.WriteArray(np.reshape(self.answers,self.dims))
        
        band = None
        dest = None
        src = None
        pass


    def displayTif(self, filepath=None, outfile=None):

        if filepath is None:
            filepath = self.output

        tif = gdal.Open(filepath)
        band = tif.GetRasterBand(1)
        img = band.ReadAsArray()

        cmap = plt.cm.get_cmap('PiYG',5)
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist[0] = (0,0,0,0) #Border pixels
        cmaplist[1] = (1.0,1.0,1.0,1.0) #Cloud
        cmaplist[2] = (.5,.5,0,0) #Desert
        cmaplist[3] = (0,0,1.0,1.0) #Water
        cmaplist[4] = (0,.5,0,1.0) #Vegetation
        cmap = cmap.from_list('Custom cmap',cmaplist,cmap.N)

        imgplot = plt.imshow(img, interpolation="none",cmap = cmap)

        def format_coord(x, y):
            col = int(x+0.5)
            row = int(y+0.5)
            z = img[row,col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)

        plt.gca().format_coord = format_coord
        plt.axis('off')
        if outfile != None:
            plt.savefig(outfile,bbox_inches='tight')
        plt.show() 
