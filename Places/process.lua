require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'loadcaffe'
require 'xlua'
require 'cudnn'
require 'cutorch'
require 'cunn'
local ffi = require 'ffi'
require 'paths'


--torch.setdefaulttensortype("torch.CudaTensor")

local M={}

-------------------------------------------------------------------------
-------------------------------------------------------------------------
--LOADS THE NETWORK MODEL-------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function loadnetwork()

	if opt.network == '' then
		print('loading previously trained network (from Caffe)')
		-- this will load the network and print it's structure
		--load AlexNet
		network = loadcaffe.load('/data/durall/CNN/deploy.prototxt', '/data/durall/CNN/bvlc_alexnet.caffemodel', 'cudnn')
		network:remove(24) -- remomve cudnn.SoftMax
		network:add(nn.Linear(1000,10))
		network:add(nn.LogSoftMax())

		--load VGGNet
		--model = loadcaffe.load('/data/durall/CNN/NETWORK1/Places_CNDS/deploy.prototxt', '/data/durall/CNN/NETWORK1/Places_CNDS/8conv3fc_DSN.caffemodel')

	else
		print('reloading previously trained network from your (t7)')
		network = torch.load(opt.network)
	end

	--convert the model into cudnn
	cudnn.convert(network,cudnn)   
	cudnn.fastest = true
	cudnn.benchmark = true
	
	-- Use a deterministic convolution implementation
	network:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
	print(network)

	model = network:cuda()

	--[[model = nn.Sequential()
	function nn.Copy.updateGradInput() end    
	model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'):cuda())
	model:add(network)]]
	print(model)

end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--INITIALIZES LAST LAYERS AND FREEZES THE OTHERS ------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function initialization()

	print('initialization')
	
	for i,m in pairs(model:listModules()) do

		--initialize weight (fc6 is layer 16)
		if i > 16  then
			if (opt.initializeAll) then
				print('initialized layer: ' .. i)
				m:reset() --random initialization of weights and bias

				if (opt.saveWeight) and (i==17 or i==20 or i==23 or i==24) then
					print("new weights saved on t7 file")
			
					--converstion to double to work without cuda library later on
					weightsDouble = model.modules[i].weight	
					weightsDouble = weightsDouble:view(weightsDouble:nElement())		
					weightsDouble = nn.utils.recursiveType(weightsDouble, 'torch.DoubleTensor')				
					local filename = paths.concat(opt.save, i .. "weight.t7")
					torch.save(filename, weightsDouble)

				end
			end

		else
			print("frozen layer: " .. i)
			--avoid to backprop through these layers
			m.updateGradInput = function(self,i,o) end -- for the gradInput
			m.accGradParameters = function(self,i,o) end -- for freezing the parameters	
		end
	end
	collectgarbage()
end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--LOAD THE DATA FROM THE FILE--------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function loadData(option,batchSize,num)
	--print('loading Place data to Torch')
	
	--Create the table of data according to
	data1 = {}
	label1 = {}


	if option == 1 then
		--train dataset
		cacheTrainFile = '/data/durall/CNN/dataset/places10DataTrain.t7'
		f = torch.load(cacheTrainFile,'ascii')
			
	else
		--test dataset
		cacheTrainFile = '/data/durall/CNN/dataset/places10DataTest.t7'
		f = torch.load(cacheTrainFile,'ascii')

	end

	startWith = 1+(num*batchSize)
	endWith = batchSize+(num*batchSize)

	if option == 1 then
		if endWith > randValTrain:size(1) then

			startWith = randValTrain:size(1) - batchSize
			endWith = randValTrain:size(1)
		end
	else
		if endWith > randValTest:size(1) then
		
			startWith = randValTest:size(1) - batchSize
			endWith = randValTest:size(1)	
		end
	end


	for i= startWith, endWith do
		
		if option == 1 then
			pathImage = ffi.string(f.imagePath[randValTrain[i]]:data())
			table.insert(label1,  f.imageClass[randValTrain[i]])
		else
			pathImage = ffi.string(f.imagePath[randValTest[i]]:data())
			table.insert(label1,  f.imageClass[randValTest[i]])
		end

		local img = image.load(pathImage,3)
		img = image.scale(img, 227, 227, bilinear)
		table.insert(data1, img)		
	end

	--Reshape the tables to Tensors
	label = torch.Tensor(#label1)
	data = torch.Tensor(#data1,3,227,227)

	for i=1, #data1 do
	    data[i] = data1[i]
	    label[i] = label1[i]
	end


	if option == 1 then
		trainData = {
			data = data,
			labels = label:cuda()
			}
		--print(trainData)
	else
		testData = {
			data = data,
			labels = label:cuda()
			}
		--print(testData)
	end
	collectgarbage()

end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--PREPROCESSING THE DATA-------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function preprocess(option)

	-- preprocess trainSet
	normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7)):float()

	if option==1 then 
		--print('preprocessing data (color space + normalization) training')  
		-- preprocess requires floating point
		trainData.data = trainData.data:float()
	

		-- data augmentation
		if opt.augmentation then
			--print('data augmenatation')	
			dataAugmentation()
		end

		for i = 1,trainData.data:size(1) do
		   -- rgb -> yuv
		   local rgb = trainData.data[i]
		   local yuv = image.rgb2yuv(rgb)
		   -- normalize y locally:
		   yuv[1] = normalization(yuv[{{1}}])
		   trainData.data[i] = yuv
		end
		-- normalize u globally:
		mean_u = trainData.data[{ {},2,{},{} }]:mean()
		std_u = trainData.data[{ {},2,{},{} }]:std()
		trainData.data[{ {},2,{},{} }]:add(-mean_u)
		trainData.data[{ {},2,{},{} }]:div(-std_u)
		-- normalize v globally:
		mean_v = trainData.data[{ {},3,{},{} }]:mean()
		std_v = trainData.data[{ {},3,{},{} }]:std()
		trainData.data[{ {},3,{},{} }]:add(-mean_v)
		trainData.data[{ {},3,{},{} }]:div(-std_v)

	else 
		--print('preprocessing data (color space + normalization) testing')  
		testData.data = testData.data:float()
		-- preprocess testSet
		for i = 1,testData.data:size(1) do
		   -- rgb -> yuv
		   local rgb = testData.data[i]
		   local yuv = image.rgb2yuv(rgb)
		   -- normalize y locally:
		   yuv[{1}] = normalization(yuv[{{1}}])
		   testData.data[i] = yuv
		end
		-- normalize u globally:
		mean_u = testData.data[{ {},2,{},{} }]:mean()
		std_u = testData.data[{ {},2,{},{} }]:std()
		testData.data[{ {},2,{},{} }]:add(-mean_u)
		testData.data[{ {},2,{},{} }]:div(-std_u)
		-- normalize v globally:
		mean_v = testData.data[{ {},3,{},{} }]:mean()
		std_v = testData.data[{ {},3,{},{} }]:std()
		testData.data[{ {},3,{},{} }]:add(-mean_v)
		testData.data[{ {},3,{},{} }]:div(-std_v)
	end
	collectgarbage()

end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--DATA AUGMENTATION-------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function dataAugmentation()

	--define how many of extra images (with augmentation) will have
	local numCrop = math.ceil(1*trainData.data:size(1))
	local numRotate = math.ceil(1*trainData.data:size(1))
	local numTrans = math.ceil(1*trainData.data:size(1))
	local numFlip = math.ceil(1*trainData.data:size(1))


	newData = trainData.data:clone()
	local sizeImg1 = trainData.data:size(3)
	local sizeImg2 = trainData.data:size(4)
	local sizeImg1Crop = math.ceil(sizeImg1/2)
	local sizeImg2Crop = math.ceil(sizeImg2/2)

	--print("numCrop: ".. numCrop)
	--three different crops are applied
	for i=1, numCrop  do
		newData = image.crop(trainData.data[i], "c", sizeImg1Crop, sizeImg2Crop)		
		newData = newData:resize(1,3,sizeImg1,sizeImg2)
		trainData.data = torch.cat(trainData.data,newData,1)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)

		newData = image.crop(trainData.data[i], "tl", sizeImg1Crop, sizeImg2Crop)		
		newData = newData:resize(1,3,sizeImg1,sizeImg2)
		trainData.data = torch.cat(trainData.data,newData,1)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)

		newData = image.crop(trainData.data[i], "br", sizeImg1Crop, sizeImg2Crop)		
		newData = newData:resize(1,3,sizeImg1,sizeImg2)
		trainData.data = torch.cat(trainData.data,newData,1)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)
	end
	--print("numRotate: ".. numRotate)
	--rotates image src by theta radians.
	for i=1, numRotate  do
		newData = image.rotate(trainData.data[i],2)
		newData = newData:resize(1,3,sizeImg1,sizeImg2)
		trainData.data = torch.cat(trainData.data,newData,1)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)
	end
	--print("numTrans: ".. numTrans)
	--translates image src by x pixels horizontally and y pixels vertically.
	for i=1, numTrans  do
		newData = image.translate(trainData.data[i],5,5)
		newData = newData:resize(1,3,sizeImg1,sizeImg2)
		trainData.data = torch.cat(trainData.data,newData,1)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)
	end
	--print("numFlip:".. numFlip)
	--flips image src vertically (upsize<->down).
	for i=1, numFlip  do

		newData = image.vflip(trainData.data[i])
		newData = newData:resize(1,3,sizeImg1,sizeImg2)
		trainData.data = torch.cat(trainData.data,newData,1)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)
	end
	collectgarbage()
end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--TRAINING---------------------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function train()

	--used in activations function
	activation1 = {}
	activation2 = {}
	activation3 = {}
	activation4 = {}
	fisrtTime = true


	-- epoch tracker
	epoch = epoch or 1

	--randValTrain = torch.randperm(280267)
	randValTrain = torch.randperm(135900)


	local inputs = {}
	local targets = {}

	-- do one epoch
	print('training set:')
	print("online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	
	local loopValue = torch.floor(randValTrain:size(1)/opt.batchSize)

	for g = 0, loopValue do

		xlua.progress(g, loopValue)

		-- number 1 is for training
		loadData(1,opt.batchSize,g)
		preprocess(1)
		trainData.data = trainData.data:cuda()

		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)

			-- just in case:
			collectgarbage()

			-- get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end

			 -- reset gradients
			gradParameters:zero()
			-- f is the average of all criterions
			local f = 0
			-- evaluate function for complete mini batch

			for i = 1,trainData.data:size(1) do
			--print(i)
				-- estimate f	
				local output = model:forward(trainData.data[i])
				local err = criterion:forward(output, trainData.labels[i])
				f = f + err

				-- estimate df/dW
				local df_do = criterion:backward(output, trainData.labels[i])
				model:backward(trainData.data[i], df_do)
				output = output:double()

				-- update confusion
				confusion:add(output, trainData.labels[i])

				--activations(trainData.labels[i],t,0)

			end
			-- normalize gradients and f(X)
			gradParameters:div(trainData.data:size(1))
			f = f/trainData.data:size(1)

			-- return f and df/dX
			return f,gradParameters
		end

		-- optimize on current mini-batch
		if opt.optimization == 'CG' then
			config = config or {maxIter = opt.maxIter}
			optim.cg(feval, parameters, config)

		elseif opt.optimization == 'LBFGS' then
			config = config or {learningRate = opt.learningRate, maxIter = opt.maxIter, nCorrection = 10}
			optim.lbfgs(feval, parameters, config)

		elseif opt.optimization == 'SGD' then
			config = config or {learningRate = opt.learningRate, weightDecay = opt.weightDecay, momentum = opt.momentum, learningRateDecay = 5e-7}
			optim.sgd(feval, parameters, config)

		elseif opt.optimization == 'ASGD' then
			config = config or {eta0 = opt.learningRate, t0 = trsize * opt.t0}
			_,_,average = optim.asgd(feval, parameters, config)

		else
			error('unknown optimization method')
		end
	end

	-- print confusion matrix
	print(confusion)
	trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	confusion:zero()

	-- save/log current net
	local filename = paths.concat(opt.save, 'Places10Caffe.t7')	
	os.execute('mkdir -p ' .. sys.dirname(filename))
	print('saving network to '..filename)
	torch.save(filename, model)



	-- next epoch
	epoch = epoch + 1
	collectgarbage()
end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--TESTING----------------------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function test()

	--used in activations function
	activation1 = {}
	activation2 = {}
	activation3 = {}
	activation4 = {}
	fisrtTime = true


	-- averaged param use?
	if average then
		cachedparams = parameters:clone()
		parameters:copy(average)
	end

	randValTest = torch.randperm(15100)

	local loopValue = torch.floor(randValTest:size(1)/opt.batchSize)

	for g = 0, loopValue do

		xlua.progress(g, loopValue)

		-- number 1 is for training
		loadData(2,opt.batchSize,g)
		preprocess(2)
		testData.data = testData.data:cuda()

		-- test over given dataset
		for t = 1, testData.data:size(1) do

			-- get new sample
			local input = testData.data[t]
			local target = testData.labels[t]

			-- test sample
			local pred = model:forward(input)
			--activations(target,t,1)

			-- Get the top 5 class indexes and probabilities
			--local N=5
			--local probs, indexes = pred:topk(N, true, true)
			--for n=1,N do
				--print(probs[n], indexes[n])
				
			--end
			--print('')

			confusion:add(pred, target)
			collectgarbage()
		end
	end

	-- print confusion matrix
	print(confusion)
	precRecall()
	testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
	confusion:zero()

	-- averaged param use?
	if average then
		-- restore parameters
		parameters:copy(cachedparams)
	end
	collectgarbage()

end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--CALCULATE PRECIONS AND RECALL------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function precRecall()
	local tp, fn, fp
	tp  = torch.diag(confusion.mat):resize(1,confusion.nclasses )
	fn = (torch.sum(confusion.mat,2)-torch.diag(confusion.mat)):t()
	fp = torch.sum(confusion.mat,1)-torch.diag(confusion.mat)
	tp = tp:float()
	fn = fn:float()
	fp = fp:float()
	prec = torch.cdiv(tp,tp+fp)
	recall = torch.cdiv(tp,tp+fn)
	print("precicion:")	
	print(prec)
	print("recall:")
	print(recall)
end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--ACTIVTIONS VALUES ARE SAVED IN A FILE----------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function activations(labelTarget,t,option)

	--num stnads for the amount of pictures
	num=1000

	if t<=num then
		--loop for save the 4 activation layers
		for i=1, 4 do
			--activation from training or testing (if it's from training we want only one activation layer)
			if option == 0 then 
				i=5
				lay=16
			else
				--activations from layer 12 
				if i==1 then
					lay = 12
				--activations from layer 15
				elseif i==2 then
					lay = 15
				--activations from layer 19
				elseif i==3 then
					lay = 19
				--activations from layer 23
				else
					lay = 23
				end
			end

			newActivation = {}
		
			newActivation = model:get(lay).output
			newActivation = newActivation:view(newActivation:nElement())
			newActivation = nn.utils.recursiveType(newActivation, 'torch.DoubleTensor')
			newActivation = newActivation:resize(1,newActivation:size(1))

			--print(newActivation:size())

			if fisrtTime then
				
				--print(newActivation:size())

				--activations from layer 12 
				if i==1 then
					activation1.data = newActivation:clone()
					activation1.labels = torch.Tensor(1):fill(labelTarget)
				--activations from layer 15 
				elseif i==2 then
					activation2.data = newActivation:clone()
					activation2.labels = torch.Tensor(1):fill(labelTarget)
				--activations from layer 19 
				elseif i==3 then
					activation3.data = newActivation:clone()
					activation3.labels = torch.Tensor(1):fill(labelTarget)
				--activations from layer 23 
				else
					activation4.data = newActivation:clone()
					activation4.labels = torch.Tensor(1):fill(labelTarget)
					fisrtTime = false					
				end

				
			else

				--print(newActivation:size())

				--activations from layer 12 
				if i==1 then
					activation1.data = torch.cat(activation1.data,newActivation,1)
					activation1.labels = torch.cat(activation1.labels,torch.Tensor(1):fill(labelTarget),1)
				--activations from layer 15 
				elseif i==2 then
					activation2.data = torch.cat(activation2.data,newActivation,1)
					activation2.labels = torch.cat(activation2.labels,torch.Tensor(1):fill(labelTarget),1)
				--activations from layer 19 
				elseif i==3 then
					activation3.data = torch.cat(activation3.data,newActivation,1)
					activation3.labels = torch.cat(activation3.labels,torch.Tensor(1):fill(labelTarget),1)
				--activations from layer 23 
				else
					activation4.data = torch.cat(activation4.data,newActivation,1)
					activation4.labels = torch.cat(activation4.labels,torch.Tensor(1):fill(labelTarget),1)
				end

			end
		
			if t == num then 
				local filename = paths.concat(opt.save, i ..'activation.t7')
			
				--activations from layer 12 
				if i==1 then
					torch.save(filename, activation1, 'ascii')
				--activations from layer 15 
				elseif i==2 then
					torch.save(filename, activation2, 'ascii')
				--activations from layer 19 
				elseif i==3 then
					torch.save(filename, activation3, 'ascii')
				--activations from layer 23 
				else
					torch.save(filename, activation4, 'ascii')
				end
			end
		end

	end
	collectgarbage()
	--print(activation1)
	--print(activation2)
	--print(activation3)
	--print(activation4)

end

M.loadnetwork = loadnetwork
M.initialization = initialization
M.loadData = loadData
M.dataAugmentation = dataAugmentation
M.preprocess = preprocess
M.train = train
M.test = test
M.activations = activations
M.precRecall = precRecall

return M
