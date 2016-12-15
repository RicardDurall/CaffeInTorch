require 'paths'
local sys = require 'sys'
local ffi = require 'ffi'
local M = {}

local function createTensor(dir)

	classes = {'airport_terminal','aqueduct','bookstore','bridge','castle','cathedral','cemetery','gas_station','harbor','highway','igloo','jail_cell','lighthouse','office_building','palace','parking_lot','residential_neighborhood','skyscraper','slum','snowfield','stadium','subway_station/platform','supermarket','swimming_pool/outdoor','windmill'}
	maxLength  = -1
	local imageClasses = {}
	local imagePaths = {}
	local imageClassesMix = {}
	local imagePathsMix = {}
	cont = 0

	for i=1,#classes do 

		local dir2 = dir .. '/' .. classes[i]
		print(dir2)
		local f = io.popen(' find -L '  ..dir2 .. ' -iname "*.jpg"')

		while true do 
			local line = f:read('*line')
			if not line then break end 

			local fileName = paths.basename(line)
			local path = dir2 .. '/' .. fileName
			table.insert(imageClasses, i)
			table.insert(imagePaths, path)
			maxLength = math.max(maxLength, #path+1)
			cont = cont +1

	   	 end
		f:close()
		print(cont)
	end
	print("Toatal:")
	print(cont)
	--print(imageClasses)
	--print(imagePaths)

	--shuffle the whole tensor
	shuffle = torch.randperm(cont)

	
	for i =1, cont do

		table.insert(imageClassesMix, imageClasses[shuffle[i]])
		table.insert(imagePathsMix, imagePaths[shuffle[i]])
	end

	--print(imageClassesMix[34])
	--print(imagePathsMix[34])

	return imagePathsMix, imageClassesMix
end


function trainTest(imgPath, imgClass)

	--print(imgPath:size(1))
	--print(imgClass:size(1))

	local numberTest = math.ceil(0.1*cont)
	local numberTrain = cont - numberTest

	print("#train")
	print(numberTrain)
	print("#test:")
	print(numberTest)

	imagePathTrain={}
	imagePathTest={}
	imageClassTrain={}
	imageClassTest={}


	for i=1, cont do

		if i <= numberTest then
			table.insert(imageClassTest, imgClass[i])
			table.insert(imagePathTest, imgPath[i])
		else
			table.insert(imageClassTrain, imgClass[i])
			table.insert(imagePathTrain, imgPath[i])
			
		end

	end


	--print(imageClassTrain[298])
	--print(imagePathTrain[298])

	local imgClassTestFinal = torch.LongTensor(imageClassTest)
	local imgClassTrainFinal = torch.LongTensor(imageClassTrain)


	local imgPathTestFinal = torch.CharTensor(#imagePathTest, maxLength):zero()
	local imgPathTrainFinal = torch.CharTensor(#imagePathTrain, maxLength):zero()
	

	for i,paths in ipairs(imagePathTest) do
		ffi.copy(imgPathTestFinal[i]:data(), paths)
	end
	for i,paths in ipairs(imagePathTrain) do
		ffi.copy(imgPathTrainFinal[i]:data(), paths)
	end

	return imgPathTrainFinal, imgClassTrainFinal, imgPathTestFinal, imgClassTestFinal


end

function M.ConvertToTensor(dataFile, cacheTrainFile, cacheTestFile)

	local imagePath = torch.CharTensor()
	local imageClass = torch.LongTensor()

	local imgPath, imgClass = createTensor(dataFile)
	local imgPath1, imgClass1, imgPath2, imgClass2 = trainTest(imgPath, imgClass)

	local dataTrain = {

		imagePath = imgPath1,
        	imageClass = imgClass1,
    	}

	local dataTest = {

		imagePath = imgPath2,
        	imageClass = imgClass2,
    	}


	print('save the images to ' .. cacheTrainFile)
	print(dataTrain)
	torch.save(cacheTrainFile, dataTrain,'ascii')
	print('save the images to ' .. cacheTestFile)
	print(dataTest)
	torch.save(cacheTestFile, dataTest,'ascii')

end


return M

















