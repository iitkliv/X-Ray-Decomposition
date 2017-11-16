import dicom
import os
import numpy
from matplotlib import pyplot, cm
import cv2
import numpy as np
from matplotlib import pyplot as plt

#####################################
fname='/home/siplab/test1.txt'
with open(fname) as f:
    content = f.readlines()

content = [x.strip() for x in content]

#####################################
a=0
for path in content:
	try:
		PathDicom = path
		lstFilesDCM = []  # create an empty list
		for dirName, subdirList, fileList in os.walk(PathDicom):
		    for filename in fileList:
			if ".dcm" in filename.lower():  # check whether the file's DICOM
			    lstFilesDCM.append(os.path.join(dirName,filename))

		lstFilesDCM=sorted(lstFilesDCM)
		# Get ref file

		listInstanceNum=[]
		for i in lstFilesDCM:
			RefDs = dicom.read_file(i)
			listInstanceNum.append(int(RefDs.InstanceNumber))

		lstFilesDCM = [x for _,x in sorted(zip(listInstanceNum,lstFilesDCM))]

		RefDs = dicom.read_file(lstFilesDCM[0])

		# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
		ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

		# Load spacing values (in mm)
		ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

		x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
		y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
		z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
		import matplotlib
		ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

		# loop through all the DICOM files
		for filenameDCM in lstFilesDCM:
		    # read the file
		    ds = dicom.read_file(filenameDCM)
		    # store the raw image data
		    ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array  

		sum=0

		matplotlib.image.imsave('/home/siplab/test1.png',ArrayDicom[:,:,60])
		###########################################################################################

		img = cv2.imread('/home/siplab/test1.png',0)
		ret,thresh1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
		#th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
		#    cv2.THRESH_BINARY,11,2)
		#th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
		#    cv2.THRESH_BINARY,11,2)
		#thresh1=th2
		cv2.imwrite("/home/siplab/test1_binary.jpg",thresh1)
		QR = cv2.imread('/home/siplab/test1_binary.jpg', 0) 
		mask = np.zeros(QR.shape,np.uint8) 

		contours, hierarchy = cv2.findContours(QR,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		best = 0
		maxsize = 0
		count = 0
		for cnt in contours:
		    if cv2.contourArea(cnt) > maxsize :
			maxsize = cv2.contourArea(cnt)
			best = count
		    count = count + 1

		x,y,w,h = cv2.boundingRect(contours[best])
		abc=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

		abc=img[y:y+h,x:x+w]
		cv2.imwrite("/home/siplab/test1_binary_cropped.jpg",abc)
		##############################################################################################
		array=ArrayDicom[y:y+h,x:x+w,:]
		new=numpy.zeros((array.shape[2],array.shape[1]))
		for i in range(0,array.shape[2]):
			for j in range(0,array.shape[1]):
				for k in range(0,array.shape[0]):
					sum=sum+array[k,j,i]
				new[i,j]=sum/array.shape[0]
				sum=0

	
		p1=os.path.join('/home/siplab/50_images_data/',str(4*a+1)+'.png')
		matplotlib.image.imsave(p1,new,cmap='gray')

		for i in range(0,array.shape[2]):
			for j in range(0,array.shape[1]):
				for k in range(0,array.shape[0]/3):
					sum=sum+array[k,j,i]
				new[i,j]=sum/array.shape[0]
				sum=0

		p2=os.path.join('/home/siplab/50_images_data/',str(4*a+2)+'.png')
		matplotlib.image.imsave(p2,new,cmap='gray')

		for i in range(0,array.shape[2]):
			for j in range(0,array.shape[1]):
				for k in range(array.shape[0]/3+1,2*array.shape[0]/3):
					sum=sum+array[k,j,i]
				new[i,j]=sum/array.shape[0]
				sum=0

		p3=os.path.join('/home/siplab/50_images_data/',str(4*a+3)+'.png')
		matplotlib.image.imsave(p3,new,cmap='gray')

		for i in range(0,array.shape[2]):
			for j in range(0,array.shape[1]):
				for k in range(2*array.shape[0]/3+1,array.shape[0]):
					sum=sum+array[k,j,i]
				new[i,j]=sum/array.shape[0]
				sum=0

		p4=os.path.join('/home/siplab/50_images_data/',str(4*a+4)+'.png')
		matplotlib.image.imsave(p4,new,cmap='gray')
		a=a+1

	
	except ValueError:
        	print 'Invalid value!'



