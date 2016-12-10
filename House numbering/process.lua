require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'loadcaffe'
require 'xlua'
require 'cudnn'
require 'cutorch'
require 'cunn'

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
	   model = loadcaffe.load('deploy.prototxt', 'bvlc_alexnet.caffemodel', 'cudnn')
	   model:remove(24) -- remomve cudnn.SoftMax
	   model:add(nn.Linear(1000,10))
	   model:add(nn.LogSoftMax())

	else
	   print('reloading previously trained network from you (t7)')
	   model = torch.load(opt.network)
	end

end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--INITIALIZES LAST LAYERS AND FREEZES THE OTHERS ------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function initialization()

	print('initialization')
	--initialize weight (fc6 is layer 18)
	for i,m in pairs(model:listModules()) do

		if i > 16 and (opt.initializeAll) then
			print('initialized layer: ' .. i)
			m:reset() --random initialization of weights and bias

			if (opt.saveWeight) and (i==17 or i==20 or i==23 or i==24) then
				print("new weights saved on t7 file")
			
				--converstion to double to work without cuda library later on
				weightsDouble = model.modules[i].weight			
				weightsDouble = nn.utils.recursiveType(weightsDouble, 'torch.DoubleTensor')
				local filename = paths.concat(opt.save, i .. "weight.t7")
				torch.save(filename, weightsDouble)
				collectgarbage()

			end

		else
			--avoid to backprop through these layers
			m.updateGradInput = function(self,i,o) end -- for the gradInput
			m.accGradParameters = function(self,i,o) end -- for freezing the parameters	
		end
	end
end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--LOAD THE DATA FROM THE FILE--------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function loadData()

	print('loading data to Torch')
	numberHousing()
	oxford()

end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--LOAD THE DATA FROM HOUSENUMBER--------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function numberHousing()

	print('loading House number')


	trsize = 73257
	tesize = 26032

	www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/housenumbers/'
	train_file = 'dataset/train_32x32.t7'
	test_file = 'dataset/test_32x32.t7'

	if not paths.filep(train_file) then
	   os.execute('wget ' .. www .. train_file)
	end
	if not paths.filep(test_file) then
	   os.execute('wget ' .. www .. test_file)
	end


	loaded = torch.load(train_file,'ascii')
	trainData = {
	   data = loaded.X:transpose(3,4),
	   labels = loaded.y[1],
	   size = function() return trsize end
	}

	loaded = torch.load(test_file,'ascii')
	testData = {
	   data = loaded.X:transpose(3,4),
	   labels = loaded.y[1],
	   size = function() return tesize end
	}

end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--LOAD THE DATA FROM OXFORD--------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function oxford()

	print('loading Oxford')

	totalsize = 5063
	trsize = 4500
	tesize = 563

	tfile = 'dataset/OxfordData.t7'
	loaded = torch.load(tfile,'ascii')

	print(loaded)
	--shuffle dataset
	dataSize = loaded.data:size()[1]
	shuffleIdx = torch.randperm(dataSize)
	loaded.data = loaded.data:index(1,shuffleIdx:long())
	loaded.labels = loaded.labels:index(1,shuffleIdx:long())

	trainData = {
	   data = loaded.data[{{1,trsize}}])
	   labels = loaded.data[{{1,trsize}}]),
	   size = function() return trsize end
	}
	testData = {
	   data = loaded.data[{{trsize+1,totalsize}}])
	   labels = loaded.data[{{trsize+1,totalsize}}]),
	   size = function() return tesize end
	}
	print(trainData)
	print(testData)

end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--PREPROCESSING THE DATA-------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function preprocess()

	print('preprocessing data (color space + normalization)')   

	-- preprocess requires floating point
	trainData.data = trainData.data:float()
	testData.data = testData.data:float()

	-- data augmentation
	if opt.augmentation then
		print('data augmenatation')	
		dataAugmentation()
	end

	-- preprocess trainSet
	normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7)):float()
	for i = 1,trainData:size() do
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

	-- preprocess testSet
	for i = 1,testData:size() do
	   -- rgb -> yuv
	   local rgb = testData.data[i]
	   local yuv = image.rgb2yuv(rgb)
	   -- normalize y locally:
	   yuv[{1}] = normalization(yuv[{{1}}])
	   testData.data[i] = yuv
	end
	-- normalize u globally:
	testData.data[{ {},2,{},{} }]:add(-mean_u)
	testData.data[{ {},2,{},{} }]:div(-std_u)
	-- normalize v globally:
	testData.data[{ {},3,{},{} }]:add(-mean_v)
	testData.data[{ {},3,{},{} }]:div(-std_v)

end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--DATA AUGMENTATION-------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function dataAugmentation()

	--define how many of extra images (with augmentation) will have
	local numCrop = math.ceil(0.1*trainData.size())
	local numRotate = math.ceil(0.3*trainData.size())
	local numTrans = math.ceil(0.01*trainData.size())
	local numFlip = math.ceil(0.01*trainData.size())


	newData = trainData.data:clone()
	local sizeImg1 = trainData.data:size(3)
	local sizeImg2 = trainData.data:size(4)
	local sizeImg1Crop = math.ceil(sizeImg1/2)
	local sizeImg2Crop = math.ceil(sizeImg2/2)

	print("numCrop: ".. numCrop)
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
	print("numRotate: ".. numRotate)
	--rotates image src by theta radians.
	for i=1, numRotate  do
		newData = image.rotate(trainData.data[i],2)
		newData = newData:resize(1,3,sizeImg1,sizeImg2)
		trainData.data = torch.cat(trainData.data,newData,1)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)
	end
	print("numTrans: ".. numTrans)
	--translates image src by x pixels horizontally and y pixels vertically.
	for i=1, numTrans  do
		newData = image.translate(trainData.data[i],5,5)
		newData = newData:resize(1,3,sizeImg1,sizeImg2)
		trainData.data = torch.cat(trainData.data,newData,1)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)
	end
	print("numFlip:".. numFlip)
	--flips image src vertically (upsize<->down).
	for i=1, numFlip  do

		newData = image.vflip(trainData.data[i])
		newData = newData:resize(1,3,sizeImg1,sizeImg2)
		trainData.data = torch.cat(trainData.data,newData,1)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)
	end

end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--TRAINING---------------------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function train(dataset)
	-- epoch tracker
	epoch = epoch or 1

	-- local vars
	local time = sys.clock()

	-- shuffle at each epoch
	shuffle = torch.randperm(trsize)

	-- do one epoch
	print('training set:')
	print("online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t = 1,dataset:size(),opt.batchSize do
		 -- disp progress
		 xlua.progress(t, dataset:size())
		-- create mini batch
		local inputs = {}
		local targets = {}
		local inputsResize = {}

		for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
			-- load new sample
			input = dataset.data[shuffle[i]]:double()
			local target = dataset.labels[shuffle[i]]
			table.insert(inputs, input)
			table.insert(targets, target)
	
		end

		inputResize = torch.Tensor(3,227,227):zero()
		for i=1, #inputs do	

			inputResize = image.scale(inputs[i],227,227)
			inputResize = inputResize:cuda()
			table.insert(inputsResize, inputResize)

		end

		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
			-- get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end

			 -- reset gradients
			gradParameters:zero()
			-- f is the average of all criterions
			local f = 0
			-- evaluate function for complete mini batch
			for i = 1,#inputsResize do
				-- estimate f	
				local output = model:forward(inputsResize[i])
				local err = criterion:forward(output, targets[i])
				f = f + err
			
				-- estimate df/dW
				local df_do = criterion:backward(output, targets[i])
				model:backward(inputsResize[i], df_do)
				output = output:double()

				-- update confusion
				confusion:add(output, targets[i])
			end
			-- normalize gradients and f(X)
			gradParameters:div(#inputsResize)
			f = f/#inputsResize

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

	-- time taken
	time = sys.clock() - time
	time = time / dataset:size()
	print("time to learn 1 sample = " .. (time*1000) .. 'ms')

	-- print confusion matrix
	print(confusion)
	trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	confusion:zero()

	-- save/log current net
	local filename = paths.concat(opt.save, 'houseCaffe.t7')
	os.execute('mkdir -p ' .. sys.dirname(filename))
	print('saving network to '..filename)
	torch.save(filename, model)

	-- next epoch
	epoch = epoch + 1
end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--TESTING----------------------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function test(dataset)

	-- local vars
	local time = sys.clock()

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

	-- test over given dataset
	print('on testing Set:')
	for t = 1, dataset:size() do
		-- disp progress
		xlua.progress(t, dataset:size())

		-- get new sample
		local input = dataset.data[t]:double()
		local target = dataset.labels[t]
		local inputsResize = {}

		inputResize = torch.Tensor(3,227,227):zero()
		inputResize = image.scale(input,227,227)
		inputResize = inputResize:cuda()

		-- test sample
		local pred = model:forward(inputResize)
		--activations(target,t)
		confusion:add(pred, target)
		
	end
	-- timing
	time = sys.clock() - time
	time = time / dataset:size()
	print("time to test 1 sample = " .. (time*1000) .. 'ms')

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

	--num stnads for the amount of pictures
	num=1000

	if t<=num then
		--loop for save the 4 activation layers
		for i=1, 4 do
			--activations from layer 9 (conv3)--activations from layer 9 (conv3)
			if i==1 then
				lay = 9
			--activations from layer 13 (conv5)
			elseif i==2 then
				lay = 13
			--activations from layer 17 (fc6)
			elseif i==3 then
				lay = 17
			--activations from layer 20 (fc7)
			else
				lay = 20
			end

			newActivation = {}
		
			newActivation = model:get(lay).output
			newActivation = newActivation:view(newActivation:nElement())
			newActivation = nn.utils.recursiveType(newActivation, 'torch.DoubleTensor')
			newActivation = newActivation:resize(1,newActivation:size(1))

			--print(newActivation:size())

			if fisrtTime then
				
				--print(newActivation:size())

				--activations from layer 9 (conv3)--activations from layer 9 (conv3)
				if i==1 then
					activation1.data = newActivation:clone()
					activation1.labels = torch.Tensor(1):fill(labelTarget)
				--activations from layer 13 (conv5)
				elseif i==2 then
					activation2.data = newActivation:clone()
					activation2.labels = torch.Tensor(1):fill(labelTarget)
				--activations from layer 17 (fc6)
				elseif i==3 then
					activation3.data = newActivation:clone()
					activation3.labels = torch.Tensor(1):fill(labelTarget)
				--activations from layer 20 (fc7)
				else
					activation4.data = newActivation:clone()
					activation4.labels = torch.Tensor(1):fill(labelTarget)
					fisrtTime = false
				end

				
			else

				--print(newActivation:size())

				--activations from layer 9 (conv3)--activations from layer 9 (conv3)
				if i==1 then
					activation1.data = torch.cat(activation1.data,newActivation,1)
					activation1.labels = torch.cat(activation1.labels,torch.Tensor(1):fill(labelTarget),1)
				--activations from layer 13 (conv5)
				elseif i==2 then
					activation2.data = torch.cat(activation2.data,newActivation,1)
					activation2.labels = torch.cat(activation2.labels,torch.Tensor(1):fill(labelTarget),1)
				--activations from layer 17 (fc6)
				elseif i==3 then
					activation3.data = torch.cat(activation3.data,newActivation,1)
					activation3.labels = torch.cat(activation3.labels,torch.Tensor(1):fill(labelTarget),1)
				--activations from layer 20 (fc7)
				else
					activation4.data = torch.cat(activation4.data,newActivation,1)
					activation4.labels = torch.cat(activation4.labels,torch.Tensor(1):fill(labelTarget),1)
				end

				--print(torch.Tensor(1):fill(labelTarget))

			end
		
			if t == num then 
				local filename = paths.concat(opt.save, i ..'activation.t7')
			
				--activations from layer 9 (conv3)--activations from layer 9 (conv3)
				if i==1 then
					torch.save(filename, activation1, 'ascii')
				--activations from layer 13 (conv5)
				elseif i==2 then
					torch.save(filename, activation2, 'ascii')
				--activations from layer 17 (fc6)
				elseif i==3 then
					torch.save(filename, activation3, 'ascii')
				--activations from layer 20 (fc7)
				else
					torch.save(filename, activation4, 'ascii')
				end
			end
		end

	end

	--print(activation1)
	--print(activation2)
	--print(activation3)
	--print(activation4)

end

M.loadnetwork = loadnetwork
M.initialization = initialization
M.loadData = loadData
M.numberHousing = numberHousing
M.oxford = oxofrd
M.dataAugmentation = dataAugmentation
M.preprocess = preprocess
M.train = train
M.test = test
M.activations = activations
M.precRecall = precRecall

return M
