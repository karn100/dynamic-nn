from training.trainer import Trainer
from layers.mlp import MLP
from datasets.data_sets import load_mnist
from optimizers.momentum import MomentumwithNAG
from training.losses import cross_entropy_loss
from training.metrics import accuracy

train_loader,test_loader,in_dims,num_classes = load_mnist()

model = MLP(in_dims,[256,256],num_classes)
optimizer = MomentumwithNAG(model.parameters(),lr=0.01,momentum=0.9)
trainer = Trainer(model,optimizer,cross_entropy_loss,metric=accuracy,device="cpu")

trainer.train(train_loader,test_loader,epochs=5)