require 'torch'
require 'nn'
require 'optim'
require 'cudnn'
require 'cunn'

local process = require 'process'

----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('Places Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-1, 'learning rate at t=0')
cmd:option('-learningRateDecay', 5e-7, 'learning rate decay')
cmd:option('-batchSize', 128, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 5e-4, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-threads', 2, 'nb of threads to use')
cmd:option('-initializeAll', false, 'initialize last layers')
cmd:option('-augmentation', false, 'load augmentation data')

control = 0


cmd:text()
opt = cmd:parse(arg)


-- fix seed
torch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)
print('set # of threads to ' .. opt.threads)

----------------------------------------------------------------------
-- define model to train
--
loadnetwork()

----------------------------------------------------------------------
-- on the 10-class classification problem
--
classes = {}
for i=1,10 do
	classes[i] = tostring(i)
end

----------------------------------------------------------------------
--initialization of the weights or/and freeze them
--
initialization()
--print(model:listModules())

----------------------------------------------------------------------
-- retrieve parameters (weights) and gradients
--
parameters,gradParameters = model:getParameters()

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
criterion = nn.ClassNLLCriterion():cuda()

----------------------------------------------------------------------
-- this matrix records the current confusion across classes
--
confusion = optim.ConfusionMatrix(classes)

---------------------------------------------------------------------
-- log results to files
--
trainLogger = optim.Logger(paths.concat(opt.save, 'trainCaffe.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'testCaffe.log'))


----------------------------------------------------------------------
-- program

local lastValue=0
local flag = 0

while true do
	
	print("---------------------------------------------------------------")
	print("---------------------------------------------------------------")

	-- training function
	process.train()
	-- test function
	process.test()
	if control >= lastValue then
		lastValue = control
		flag = 0
	else
		if flag == 1 then 
			opt.learningRate = opt.learningRate/10
			print("It might have overfitting, learning decay by factor 10: " ..  opt.learningRate)
			lastValue = control
			flag = 0
		else
			flag = flag +1
		end
		
	end

	print("---------------------------------------------------------------")
	print("---------------------------------------------------------------")

end

