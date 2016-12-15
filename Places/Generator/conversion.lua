require 'torch'
require 'image'
require 'paths'
local gen = require 'gen'
local ffi = require 'ffi'



dataFile = '/data/durall/CNN/dataset/Places25/data'

--train dataset
cacheTrainFile = '/data/durall/CNN/dataset/places25DataTrain.t7'
--test dataset
cacheTestFile = '/data/durall/CNN/dataset/places25DataTest.t7'

--use this fucntions only once, when we need to create t7 with the paths
gen.ConvertToTensor(dataFile, cacheTrainFile, cacheTestFile)


--[[
f = torch.load(cacheTestFile,'ascii')
print(f)

--Create the table of data according to
data1 = {}
label1 = {}

loadSize=390



for i=1, loadSize do
	pathImage = ffi.string(f.imagePath[i]:data())
	local img = image.load(pathImage,3)
	img = image.scale(img, 227, 227, bilinear)
	table.insert(data1, img)
	table.insert(label1,  f.imageClass[i])

end

trainData = {
	data = data1,
	labels = label1 ,
	size = function() return loadSize end
	}
print (trainData)]]
