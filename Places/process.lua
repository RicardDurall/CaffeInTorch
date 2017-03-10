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
		--load AlexNet
		--imagenet pretrained
		--network = loadcaffe.load('/data/durall/CNN/deploy_imageNet.prototxt', '/data/durall/CNN/bvlc_reference_caffenet.caffemodel', 'cudnn')
		--no imagenet 
		network = loadcaffe.load('/data/durall/CNN/deploy.prototxt', '/data/durall/CNN/bvlc_alexnet.caffemodel', 'cudnn')
		network:remove(24) -- remomve cudnn.SoftMax
		network:add(nn.Linear(1000,10))
		network:add(nn.LogSoftMax())

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
	--print(network)

	model = network:cuda()
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

		--initialize weight (fc6 is layer 16) 12
		if i > 30  then
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
	activationSum = {}
	res1 = {}
	res2 = {}
	res3 = {}
	res4 = {}
	res5 = {}
	res6 = {}
	res7 = {}
	res8 = {}
	res9 = {}
	res10 = {}
	fisrtTime = true


	-- epoch tracker
	epoch = epoch or 1

	--25 randValTrain = torch.randperm(280267)
	--10 1.randValTrain = torch.randperm(135900)
	randValTrain = torch.randperm(112798)

	local inputs = {}
	local targets = {}

	-- do one epoch
	print('training set:')
	print("online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	
	local loopValue = torch.floor(randValTrain:size(1)/opt.batchSize)
	t = 0
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
				-- estimate f	
				local output = model:forward(trainData.data[i])
				local err = criterion:forward(output, trainData.labels[i])

				f = f + err

				-- estimate df/dW
				local df_do = criterion:backward(output, trainData.labels[i])
				model:backward(trainData.data[i], df_do)
				output = output:double()
				t = t+1
				-- update confusion
				confusion:add(output, trainData.labels[i])
				activations(trainData.labels[i],t)
				--summation(trainData.labels[i],false,0)
				

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
			config = config or {learningRate = opt.learningRate, weightDecay = opt.weightDecay, momentum = opt.momentum, learningRateDecay = opt.learningRateDecay}
			optim.sgd(feval, parameters, config)

		elseif opt.optimization == 'ASGD' then
			config = config or {eta0 = opt.learningRate}
			_,_,average = optim.asgd(feval, parameters, config)

		else
			error('unknown optimization method')
		end
	end

	--save values sumation
	--summation(1,true,0)

	-- print confusion matrix
	print(confusion)
	trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	confusion:zero()

	--save/log current net
	--local filename = paths.concat(opt.save, 'Places10Caffe.t7')	
	--os.execute('mkdir -p ' .. sys.dirname(filename))
	--print('saving network to '..filename)
	--torch.save(filename, model)



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
	activationSum = {}
	res1 = {}
	res2 = {}
	res3 = {}
	res4 = {}
	res5 = {}
	res6 = {}
	res7 = {}
	res8 = {}
	res9 = {}
	res10 = {}
	fisrtTime = true


	-- averaged param use?
	if average then
		cachedparams = parameters:clone()
		parameters:copy(average)
	end

	--25 randValTest = torch.randperm(31141)
	--10 1.randValTest = torch.randperm(15100)
	randValTest = torch.randperm(12534)

	local loopValue = torch.floor(randValTest:size(1)/opt.batchSize)
	i=0
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
			i = i+1

			activations(target,i)
			--summation(testData.labels[t],false,1)

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

	--save values sumation
	--summation(1,true,1)

	-- print confusion matrix
	print(confusion)
	precRecall()
	testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
	control = confusion.totalValid * 100
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
function activations(labelTarget,t)
	--size of the saved file	
	size = 100
	--num stnads for the amount of pictures
	num=1000
	print(t)
	if t<=num then
		--loop for save the 4 activation layers
		for i=1, 4 do
			--activation from training or testing 

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
					print(activation1)
				--activations from layer 15 
				elseif i==2 then
					activation2.data = newActivation:clone()
					activation2.labels = torch.Tensor(1):fill(labelTarget)
					print(activation2)
				--activations from layer 19 
				elseif i==3 then
					activation3.data = newActivation:clone()
					activation3.labels = torch.Tensor(1):fill(labelTarget)
					print(activation3)
				--activations from layer 23 
				else
					activation4.data = newActivation:clone()
					activation4.labels = torch.Tensor(1):fill(labelTarget)
					print(activation4)
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
			if t%size==0 then 

				local filename = paths.concat(opt.save, 'test/' .. t .. i ..'activationTest.t7')
			
				--activations from layer 12 
				if i==1 then
					torch.save(filename, activation1, 'ascii')
					print(activation1)
				--activations from layer 15 
				elseif i==2 then
					torch.save(filename, activation2, 'ascii')
					print(activation2)
				--activations from layer 19 
				elseif i==3 then
					torch.save(filename, activation3, 'ascii')
					print(activation3)
				--activations from layer 23 
				else
					torch.save(filename, activation4, 'ascii')
					fisrtTime = true
					print(activation4)
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

function summation(labelTarget,flag,test)
	if(flag==false) then 
		lay = 23
		newActivationSum = model:get(lay).output
		newActivationSum = newActivationSum:view(newActivationSum:nElement())
		--print(labelTarget)

		if fisrtTime then
			res1.data = newActivationSum:clone()*0
			res1.number = 0
			res2.data = newActivationSum:clone()*0
			res2.number = 0
			res3.data = newActivationSum:clone()*0
			res3.number = 0
			res4.data = newActivationSum:clone()*0
			res4.number = 0
			res5.data = newActivationSum:clone()*0
			res5.number = 0
			res6.data = newActivationSum:clone()*0
			res6.number = 0
			res7.data = newActivationSum:clone()*0
			res7.number = 0
			res8.data = newActivationSum:clone()*0
			res8.number = 0
			res9.data = newActivationSum:clone()*0
			res9.number = 0
			res10.data = newActivationSum:clone()*0
			res10.number = 0
			fisrtTime = false	
		end

		if(labelTarget ==1) then
			res1.data = res1.data + newActivationSum
			res1.number = res1.number +1
		elseif (labelTarget ==2) then
			res2.data = res2.data + newActivationSum
			res2.number = res2.number +1
		elseif (labelTarget ==3) then
			res3.data = res3.data + newActivationSum
			res3.number = res3.number +1
		elseif (labelTarget ==4) then
			res4.data = res4.data + newActivationSum
			res4.number = res4.number +1
		elseif (labelTarget ==5) then
			res5.data = res5.data + newActivationSum
			res5.number = res5.number +1
		elseif (labelTarget ==6) then
			res6.data = res6.data + newActivationSum
			res6.number = res6.number +1
		elseif (labelTarget ==7) then
			res7.data = res7.data + newActivationSum
			res7.number = res7.number +1
		elseif (labelTarget ==8) then
			res8.data = res8.data + newActivationSum
			res8.number = res8.number +1
		elseif (labelTarget ==9) then
			res9.data = res9.data + newActivationSum
			res9.number = res9.number +1
		elseif (labelTarget ==10) then
			res10.data = res10.data + newActivationSum
			res10.number = res10.number +1
		end

	elseif(flag==true) then
		res1.data = res1.data / res1.number
		actiLogger1 = optim.Logger(paths.concat(opt.save, test .. 'activation1.log'))

		res2.data = res2.data / res2.number
		actiLogger2 = optim.Logger(paths.concat(opt.save, test .. 'activation2.log'))

		res3.data = res3.data / res3.number
		actiLogger3 = optim.Logger(paths.concat(opt.save, test .. 'activation3.log'))

		res4.data = res4.data / res4.number
		actiLogger4 = optim.Logger(paths.concat(opt.save, test .. 'activation4.log'))

		res5.data = res5.data / res5.number
		actiLogger5 = optim.Logger(paths.concat(opt.save, test .. 'activation5.log'))

		res6.data = res6.data / res6.number
		actiLogger6 = optim.Logger(paths.concat(opt.save, test .. 'activation6.log'))

		res7.data = res7.data / res7.number
		actiLogger7 = optim.Logger(paths.concat(opt.save, test .. 'activation7.log'))

		res8.data = res8.data / res8.number
		actiLogger8 = optim.Logger(paths.concat(opt.save, test .. 'activation8.log'))

		res9.data = res9.data / res9.number
		actiLogger9 = optim.Logger(paths.concat(opt.save, test .. 'activation9.log'))

		res10.data = res10.data / res10.number
		actiLogger10 = optim.Logger(paths.concat(opt.save, test .. 'activation10.log'))


		for i =1, res1.data:size(1) do 
			actiLogger1:add{['% mean class accuracy (test set)'] = res1.data[i]}
			actiLogger2:add{['% mean class accuracy (test set)'] = res2.data[i]}
			actiLogger3:add{['% mean class accuracy (test set)'] = res3.data[i]}
			actiLogger4:add{['% mean class accuracy (test set)'] = res4.data[i]}
			actiLogger5:add{['% mean class accuracy (test set)'] = res5.data[i]}
			actiLogger6:add{['% mean class accuracy (test set)'] = res6.data[i]}
			actiLogger7:add{['% mean class accuracy (test set)'] = res7.data[i]}
			actiLogger8:add{['% mean class accuracy (test set)'] = res8.data[i]}
			actiLogger9:add{['% mean class accuracy (test set)'] = res9.data[i]}
			actiLogger10:add{['% mean class accuracy (test set)'] = res10.data[i]}
		end
			
	end

end

M.summation = summation
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
