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

	return model

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

		if i > 17 and (opt.initializeAll) then
			print('initialized last layers')
			m:reset() --random initialization of weights and bias

			if (opt.saveWeight) and (i==18 or i==21 or i==24 or i==25) then
				print("new weights saved on t7 file")
			
				--converstion to double to work without cuda library later on
				weightsDouble = model.modules[i].weight			
				nn.utils.recursiveType(weightsDouble, 'torch.DoubleTensor')

				local filename = paths.concat(opt.save, i .. ".t7")
				torch.save(filename, weightsDouble)
				collectgarbage()

			end

		else
			--avoid to backprop through these layers
			m.updateGradInput = function(self,i,o) end -- for the gradInput
			m.accGradParameters = function(self,i,o) end -- for freezing the parameters	
		end
	end

	return model

end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--LOAD THE DATA FROM THE FILE--------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function loadData()

	print('loading data to Torch')


	trsize = 73257
	tesize = 26032

	www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/housenumbers/'
	train_file = 'train_32x32.t7'
	test_file = 'test_32x32.t7'

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


	return trainData,testData

end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--PREPROCESSING THE DATA-------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function preprocess()

	print('data augmenatation and preprocessing data (color space + normalization)')   

	-- preprocess requires floating point
	trainData.data = trainData.data:float()
	--trainData.data = trainData.data:resize(trainData.data:size(),3,227,227)
	print(trainData.size())
	testData.data = testData.data:float()

	-- data augmentation
	if opt.augmentation then	
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

	return trainData,testData

end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--DATA AUGMENTATION-------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function dataAugmentation()

	--define how many of extra images (with augmentation) will have
	local numCrop = math.ceil(0.001*trainData.size())
	local numRotate = math.ceil(0.001*trainData.size())
	local numTrans = math.ceil(0.0001*trainData.size())
	local numFlip = math.ceil(0.0001*trainData.size())


	newData = trainData.data:clone()

	--three different crops are applied
	for i=1, numCrop  do

		newData = image.crop(trainData.data[i], "c", 20, 20)		
		newData = newData:resize(1,3,32,32)
		trainData.data = torch.cat(trainData.data,newData,1)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)
		newData = image.crop(trainData.data[i], "tl", 20, 20)		
		newData = newData:resize(1,3,32,32)
		trainData.data = torch.cat(trainData.data,newData,1)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)
		newData = image.crop(trainData.data[i], "br", 20, 20)		
		newData = newData:resize(1,3,32,32)
		trainData.data = torch.cat(trainData.data,newData,1)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)
	end
	print(numRotate)
	--rotates image src by theta radians.
	for i=1, numRotate  do
		newData = image.rotate(trainData.data[i],2)
		newData = newData:resize(1,3,32,32)
		trainData.data = torch.cat(trainData.data,newData,1)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)
	end
	print(numTrans)
	--translates image src by x pixels horizontally and y pixels vertically.
	for i=1, numTrans  do
		newData = image.translate(trainData.data[i],5,5)
		newData = newData:resize(1,3,32,32)
		trainData.data = torch.cat(trainData.data,newData,1)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)
	end
--[[
	print(numFlip)
	--flips image src vertically (upsize<->down).
	for i=1, numFlip  do
	print(i)
		newData = image.vflipl(trainData.data[i])
	print(i)
		newData = newData:resize(1,3,32,32)
	print(i)
		trainData.data = torch.cat(trainData.data,newData,1)
	print(i)
		newLabel = torch.Tensor(1):fill(trainData.labels[i])
		trainData.labels = torch.cat(trainData.labels,newLabel,1)
	end]]

print(trainData:size())


return trainData

end
-------------------------------------------------------------------------
-------------------------------------------------------------------------
--TRAINING---------------------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
function train()
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

      if input:size(2) ~= 227 then

		 inputResize = torch.Tensor(3,227,227):zero()
		 --print("num",inputs)
		 for i=1, #inputs do	

		     inputResize = image.scale(inputs[i],227,227)
		     inputResize = inputResize:cuda()
         	     table.insert(inputsResize, inputResize)

		 end
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
         config = config or {learningRate = opt.learningRate,
                             maxIter = opt.maxIter,
                             nCorrection = 10}
         optim.lbfgs(feval, parameters, config)

      elseif opt.optimization == 'SGD' then
         config = config or {learningRate = opt.learningRate,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = 5e-7}
         optim.sgd(feval, parameters, config)

      elseif opt.optimization == 'ASGD' then
         config = config or {eta0 = opt.learningRate,
                             t0 = trsize * opt.t0}
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
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over given dataset
   print('on testing Set:')
   for t = 1,dataset:size() do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- get new sample
      local input = dataset.data[t]:double()
      local target = dataset.labels[t]
      local inputsResize = {}

      if input:size(2) ~= 227 then

		 inputResize = torch.Tensor(3,227,227):zero()
		 inputResize = image.scale(input,227,227)
		 inputResize = inputResize:cuda()

      end

      -- test sample
      local pred = model:forward(inputResize)
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
end

M.dataAugmentation = dataAugmentation
M.loadnetwork = loadnetwork
M.initialization = initialization
M.preprocess = preprocess
M.loadData = loadData
M.train = train
M.test = test

return M

