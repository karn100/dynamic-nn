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
            running_loss = 0.0
            
            # batch_idx = index of current batch in which data is grouped,batch_idx = 0,1,2..(batch no.)
            # x = 100 or 1000 images or data ,in_dims = features per samples -
            # if one image has has 28 x 28 pixels = in_dims = 784, if its 3 channel RGB then 3 x 28 x 28

            for batch_idx,(x,y) in enumerate(train_loader):  
                x,y = x.to(self.device), y.to(self.device)
            
            # forward
                pred = self.model(x)  # its a short method for self.model.forward(x)
                loss = self.loss_fn(pred,y)
                running_loss += loss.item()

            #backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            avg_loss = running_loss/len(train_loader)

            log = f"[Epochs {epoch}/{epochs}] Train Loss: {avg_loss:.4f} |"

            #train_loader = Training data which is used while training(model.train()).Here, Dropout and Batchnorm are ON.
            # Val_loader = Evaluation data which is used to check performance while training to see if -
            # our model is not overfitting(memorizig data instead of learning patterns).
            # gradients are of no use here, dropout and batchnorm are also off.

            if val_loader is not None:
                val_loss,val_metric = self.evaluate(val_loader)
                log += f"Val Loss: {val_loss:.4f} |" # we used '+= here to append the val_loss in above log where we calculated train_loss
                if self.metric is not None:
                    log += f"Val_{self.metric.__name__}:{val_metric:.4f} |"
            
            if epoch % self.print_every == 0:
                elapsed = time.time() - start_time
                log += f"Time: {elapsed:.2f}s"
                print(log)
    
    def evaluate(self,loader):
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0

        with torch.no_grad():
            for x,y in loader:
                pred = self.model(x)
                loss = self.loss_fn(pred,y)
                total_loss += loss.item()
                
                if self.metric is not None:
                    total_metric += self.metric(pred,y)
                
        
        avg_loss = total_loss / len(loader)
        avg_metric = total_metric / len(loader) if self.metric is not None else None
        return avg_loss , avg_metric
    
    