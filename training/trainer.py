import torch
import time

class Trainer:
    def __init__(self,model,optimizer,loss_fin,metric = None,device = None,print_every=1):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fin
        self.metric = metric

        self.device = device or (torch.device("cuda") if torch.cuda.is_available else torch.device("cpu"))

        self.model.to(self.device)
        self.print_every = print_every
    
    def train(self,train_loader,val_loader = None,epochs = 1):
        for epoch in range(1,epochs + 1):
            start_time = time.time()
            self.model.train()
            starting_loss = 0.0
            
            # batch_idx = index of current batch in which data is grouped
            # x = (batch_idx,in_dims) --> batch_idx = 1000 images, in_dims = features per samples -
            # if one image has has 28 x 28 pixels = in_dims = 784, if its 3 channel RGB then 3 x 28 x 28

            for batch_idx,(x,y) in enumerate(train_loader):  
                x,y = x.to(self.device), y.to(self.device)
            
            # forward
                pred = self.model(x)
                loss = self.loss_fn(pred,y)
                running_loss += loss.item()

            #backward

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            avg_loss = running_loss/len(train_loader)

            log = f"[Epochs {epoch}/{epochs}] Train Loss: {avg_loss:.4f}"
            

