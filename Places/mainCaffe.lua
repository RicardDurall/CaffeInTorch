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

local process = require 'process'

----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('Oxofrd Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
cmd:option('-batchSize', 128, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 5e-4, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-threads', 1, 'nb of threads to use')
cmd:option('-initializeAll', true, 'initialize last layers too')
cmd:option('-saveWeight', false, 'save the initialized weights')
cmd:option('-augmentation', true, 'load augmentation data')

cmd:text()
opt = cmd:parse(arg)


-- fix seed
torch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)
print('set nb of threads to ' .. opt.threads)

----------------------------------------------------------------------
-- define model to train
--
loadnetwork()
model:cuda()
print(model)

----------------------------------------------------------------------
-- on the 205-class classification problem
--
classes = {}
for i=1,25 do
	classes[i] = tostring(i)
end

----------------------------------------------------------------------
--initialization of the weights or/and froze them
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

while true do
	
	print("---------------------------------------------------------------")
	print("---------------------------------------------------------------")


	-- training function
	process.train()

	-- test function
	process.test()

	print("---------------------------------------------------------------")
	print("---------------------------------------------------------------")


	--print(model.modules[1].weight[33][2][1][7])
	--print(model.modules[17].weight[33][2])

end
