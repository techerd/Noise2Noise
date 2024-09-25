#!/usr/bin/env python
#coding: utf-8

"""
Code for 1 variable, 1 climate model, annual data
"""

# Import modules

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import matplotlib
import sys
import time
from tqdm import tqdm
from contextlib import redirect_stdout
import plotly.graph_objects as go
from pathlib import Path
import gc

import random
import scipy.stats as stats
import torch.nn.functional as F
import pickle
import os
from IPython.display import display, Image

import torch.optim as optim
from torchvision.transforms import ToTensor
from torchinfo import summary
import random as rand
from sklearn.model_selection import KFold

from torch.optim.lr_scheduler import StepLR
from torch.cuda import nvtx


# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))
    

# U-Net architecture 
    
class DoubleConv_SiLU(nn.Module):
    """
    Double convolution with 
    - SiLu activation function 
    - Padding : circular on the east and west, replication on the north and south and for the time dimension
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv_SiLU, self).__init__()
        self.double_conv = nn.Sequential(
            nn.ReplicationPad3d((0,0,1,1,1,1)),
            nn.CircularPad3d((1,1,0,0,0,0)),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.SiLU(),
            nn.ReplicationPad3d((0,0,1,1,1,1)),
            nn.CircularPad3d((1,1,0,0,0,0)),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.SiLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet_SiLU(nn.Module):
    """
    U-Net structure for denoising with depth = 4
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Setting all the layers of the U-Net
        """
        super(UNet_SiLU, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.ReplicationPad3d((0,0,1,1,1,1)),
            nn.CircularPad3d((1,1,0,0,0,0)),
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=0),
            nn.Tanh(),
            nn.ReplicationPad3d((0,0,1,1,1,1)),
            nn.CircularPad3d((1,1,0,0,0,0)),
            nn.Conv3d(16, 16, kernel_size=3, padding=0),
            nn.Tanh()
        )
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = DoubleConv_SiLU(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = DoubleConv_SiLU(32, 64)
        self.pool3 = nn.MaxPool3d(2)
        
        self.bottleneck = DoubleConv_SiLU(64, 128)
        
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv_SiLU(128, 64)
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv_SiLU(64, 32)        
        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        
        self.decoder1 = nn.Sequential(
            nn.ReplicationPad3d((0,0,1,1,1,1)),
            nn.CircularPad3d((1,1,0,0,0,0)),
            nn.Conv3d(32, 8, kernel_size=3, padding=0),
            nn.Tanh()
        )
        
        self.last = nn.Sequential(
            nn.ConvTranspose3d(8, 8, kernel_size=(1,1,1)),
            nn.Tanh()
        )
                
        self.outconv = nn.Conv3d(8, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Going through the U-Net
        """
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Adjusting spatial dimensions to match before concatenation with .size()
        dec3 = self.upconv3(bottleneck) 
        dec3 = self.decoder3(torch.cat((enc3[:, :, :dec3.size(2), :dec3.size(3), :dec3.size(4)], dec3), dim=1)) 
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((enc2[:, :, :dec2.size(2), :dec2.size(3), :dec2.size(4)], dec2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((enc1[:, :dec1.size(1), :dec1.size(2), :dec1.size(3), :dec1.size(4)], dec1), dim=1))
        
        last = self.last(dec1)
        
        out = self.outconv(last)
        return out


class TrainDataset(Dataset):
    """Load data for training"""
    
    def __init__(self, variable, model, normalisation):
        """
        Create pairs of members file names
        """   
        self.variable = variable
        self.model = model
        self.normalisation = normalisation
        self.root_dir = '/scratch/globc/techer/ia/data_norm/'+normalisation+'/'+self.variable+'/'+self.model+'/' # Path with normalized data for one normalization, one variable and one model
        
        pairs_of_members = [] # All pairs of members for training
        members_eval = [26,27,41] # Define three members for the test (median change in temperature over the period for CESM2, minimal change, maximum change)

        members_nb = [i for i in range(1,51) if not i in members_eval]
        mem_val = rand.sample(members_nb, 11) # Take randomly 11 members for the validation step
        mem_training = []
        
        # Create a list of members for the training
        for member in members_nb :
            if not member in mem_val :
                mem_training.append(member)
        
        # Create pairs of members for training 
        for mem_input in mem_training:
            for mem_output in mem_training:
                if mem_input!=mem_output:
                    pairs_of_members.append([mem_input,mem_output])    
                
        self.pairs_of_members_training = np.array(pairs_of_members)
        self.members_validation = np.array(mem_val)
        
    
    def val_members(self):
        return self.members_validation

    def __len__(self):
        return self.pairs_of_members_training.shape[0]

    def __getitem__(self, item):
        """
        Load the data and convert them in PyTorch format
        """        
        pair = self.pairs_of_members_training[item]        
        
        input_member_name = self.root_dir+self.variable+'_yearly_'+self.model+'_r'+str(pair[0])+'i1p1f1.1900-2022_'+self.normalisation+'.npy' # Path of the data for the first member of the pair
        input_member = np.load(input_member_name, allow_pickle=True)[3:,:,:] # Erase the first three years to have shapes fitted to the U-Net structure and avoid losing random years
        
        # # Path of the data for the first member of the pair that is the ensemble mean when we have the same member
        if pair[0]==pair[1]:            
            target_member_name = self.root_dir+self.variable+'_yearly_'+self.model+'_r0i1p1f1.1900-2022_'+self.normalisation+'.npy'  
        else :        
            target_member_name = self.root_dir+self.variable+'_yearly_'+self.model+'_r'+str(pair[1])+'i1p1f1.1900-2022_'+self.normalisation+'.npy'
            
        target_member = np.load(target_member_name, allow_pickle=True)[3:,:,:] # Erase the first three years to have shapes fitted to the U-Net structure and avoid losing random years

        input_member = torch.tensor(input_member, dtype=torch.float32, requires_grad=True) # Convert data into PyTorch tensor
        target_member = torch.tensor(target_member, dtype=torch.float32, requires_grad=True) # Convert data into PyTorch tensor

        # Add channel dimension as U-Net expects input channels (unsqueeze) and convert tensor dtype for a use on GPU with cuda
        device = torch.set_default_device("cuda" if torch.cuda.is_available() else sys.exit("no GPU available"))
        input_member = input_member.unsqueeze(0).to(device)
        target_member = target_member.unsqueeze(0).to(device)
        
        return input_member, target_member
    
    
class ValidationDataset(Dataset):
    """Load data for validation"""

    def __init__(self, variable, model, normalisation, val_members_list):
        
        self.variable = variable
        self.model = model
        self.normalisation = normalisation
        self.root_dir = '/scratch/globc/techer/ia/data_norm/'+normalisation+'/'+self.variable+'/'+self.model+'/' # Path with normalized data for one normalization, one variable and one model
        self.member_numbers = val_members_list # List of members for the validation step
                
    def __len__(self):
        return len(self.member_numbers) 

    def __getitem__(self, item):
        """
        Load the data and convert them in PyTorch format
        """
        mem_nb = self.member_numbers[item]     
                
        input_member_name = self.root_dir+self.variable+'_yearly_'+self.model+'_r'+str(mem_nb)+'i1p1f1.1900-2022_'+self.normalisation+'.npy' # Path of the data for the first member of the pair
        input_member = np.load(input_member_name, allow_pickle=True)[3:,:,:] # Erase the first three years to have shapes fitted to the U-Net structure and avoid losing random years
        
        target_member_name = self.root_dir+self.variable+'_yearly_'+self.model+'_r0i1p1f1.1900-2022_'+self.normalisation+'.npy' # Path of the data for the second member of the pair that is the ensemble mean
        target_member = np.load(target_member_name, allow_pickle=True)[3:,:,:] # Erase the first three years to have shapes fitted to the U-Net structure and avoid losing random years

        input_member = torch.tensor(input_member, dtype=torch.float32, requires_grad=True) # Convert data into PyTorch tensor
        target_member = torch.tensor(target_member, dtype=torch.float32, requires_grad=True) # Convert data into PyTorch tensor

        # Add channel dimension as U-Net expects input channels (unsqueeze) and convert tensor dtype for a use on GPU with cuda
        device = torch.set_default_device("cuda" if torch.cuda.is_available() else sys.exit("no GPU available"))
        input_member = input_member.unsqueeze(0).to(device)
        target_member = target_member.unsqueeze(0).to(device)

        return input_member, target_member
    
    
class EvalDataset(Dataset):
    
    def __init__(self, variable, model, normalisation):
        
        self.variable = variable
        self.model = model
        self.normalisation = normalisation
        self.root_dir = '/scratch/globc/techer/ia/data_norm/'+normalisation+'/'+self.variable+'/'+self.model+'/' # Path with normalized data for one normalization, one variable and one model
        self.eval_member = np.array([26,27,41]) # List of members for the test/evaluation

    def __len__(self):
        return (len(self.eval_member)) #there is only three members for the test dataset

    def __getitem__(self, item):
        """
        Load the data and convert them in PyTorch format
        """
        mem_nb = self.eval_member[item]     
                
        input_member_name = self.root_dir+self.variable+'_yearly_'+self.model+'_r'+str(mem_nb)+'i1p1f1.1900-2022_'+self.normalisation+'.npy' # Path of the data for the first member of the pair
        input_member = np.load(input_member_name, allow_pickle=True)[3:,:,:] # Erase the first three years to have shapes fitted to the U-Net structure and avoid losing random years
        
        target_member_name = self.root_dir+self.variable+'_yearly_'+self.model+'_r0i1p1f1.1900-2022_'+self.normalisation+'.npy' # Path of the data for the second member of the pair that is the ensemble mean
        target_member = np.load(target_member_name, allow_pickle=True)[3:,:,:] # Erase the first three years to have shapes fitted to the U-Net structure and avoid losing random years

        input_member = torch.tensor(input_member, dtype=torch.float32, requires_grad=False) # Convert data into PyTorch tensor
        target_member = torch.tensor(target_member, dtype=torch.float32, requires_grad=False) # Convert data into PyTorch tensor

        # Add channel dimension as U-Net expects input channels (unsqueeze) and convert tensor dtype for a use on GPU with cuda
        device = torch.set_default_device("cuda" if torch.cuda.is_available() else sys.exit("no GPU available"))
        input_member = input_member.unsqueeze(0).to(device)
        target_member = target_member.unsqueeze(0).to(device)

        return input_member, target_member
    

class Model():
    """
    Train, validate, test and save the deep learning model
    """
    
    def __init__(self, variable, clim_model, model, normalisation, nb_epochs, batch_size, save_path):
        
        self.variable = variable
        self.clim_model = clim_model
        self.model = model
        self.normalisation = normalisation
        self.nb_epochs = nb_epochs 
        self.batch_size = batch_size
        self.save_path = save_path
        
        
    def train(self, model, device, train_loader, optimizer, epoch, criterion, lat_wgts_tensor, model_save_path):
                
        leave=False # To leave epoch loop 
        model.train() # Set the model in training mode
        self.grad_norms = {name: [] for name, _ in model.named_parameters()}  # Dictionary to store gradient norms
        train_loss = 0.0 # Initiate train loss
        
        use_amp = True # For gradient scaling and automatic mixed precision
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # Gradient scaling

        with torch.autograd.profiler.emit_nvtx(): # To have nvidia Nsight profile
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.nb_epochs} [Train]', unit='batch', leave=leave) as iterator:
                nvtx.range_push("Data loading") # Create a marker for data loading
                for i, (inputs, targets) in enumerate(iterator):
                    with torch.autocast(device_type=device, dtype=torch.float16): # Automatic mixed precision    
                        
                        nvtx.range_pop() # End of the marker for data loading for each pairs
                        nvtx.range_push("Batch" + str(i)) # Create a marker for a batch

                        # forward
                        nvtx.range_push("Forward pass") # Create a marker for the forward pass
                        outputs = model(inputs)  
                        loss = criterion(outputs*lat_wgts_tensor, targets*lat_wgts_tensor) # Compute the loss
                        nvtx.range_pop() # End of the marker for the forward pass
                    
                    # backward
                    nvtx.range_push("Backward pass") # Create a marker for the backward pass
                    scaler.scale(loss).backward() # Scale the loss for the backward pass
                    scaler.unscale_(optimizer) # Unscale the loss to save gradients
                    
                    # Compute and store the norm of gradients for each parameter
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param_norm = param.grad.norm(2).item()  # L2 norm
                                self.grad_norms[name].append(param_norm)  # Add current norm
                    # optimize
                    scaler.step(optimizer)
                    scaler.update()
                    
                    nvtx.range_pop() #  End of the marker for the backward pass
                    nvtx.range_pop() #  End of the marker for the batch

                    # Resume Amp-enabled runs with bitwise accuracy
                    checkpoint={"model":model.state_dict(),
                            "optimizer":optimizer.state_dict(),
                            "scaler":scaler.state_dict()}
                    torch.save(checkpoint, model_save_path / 'amp_checkpoint')

                    optimizer.zero_grad() # Reset the gradients

                    # Print statistics
                    train_loss += loss.item()
                    if i % max(len(iterator) // 100, 1) == 0:
                        iterator.set_postfix(
                            {
                                "Loss": f"{train_loss / (i + 1):.6f}",
                            }
                        )
                        
                    nvtx.range_push("Data loading") # Create a marker for data loading for each pairs
                nvtx.range_pop() # End of the marker for data loading
                    
        self.train_loss_list.append(train_loss / (i+1)) # Save the loss for each epoch
        
            
                    
           
                    
    def validate(self, model, device, val_loader, epoch, criterion, lat_wgts_tensor):
        
        use_amp = True # For gradient scaling and automatic mixed precision
        leave=False # To leave epoch loop 
        model.eval() # Set the model in validating mode
        val_loss = 0.0 # Initiate validation loss
        
        with tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.nb_epochs} [Val]', unit='batch', leave=leave) as iterator:
            for i, (inputs, targets) in enumerate(iterator):                    
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp): # Automatic mixed precision    

                    with torch.no_grad():
                        nvtx.range_push("Validation forward pass") # Create a marker for the forward pass
                        outputs = model(inputs)
                        nvtx.range_pop() # End of the marker for the forward pass

                    val_loss += criterion(outputs*lat_wgts_tensor, targets*lat_wgts_tensor).item() # Loss

                    if i % max(len(iterator) // 100, 1) == 0:
                        iterator.set_postfix(
                            {
                                "Loss": f"{val_loss / (i + 1):.6f}",
                            }
                    )
            
            self.val_loss_list.append(val_loss / (i+1)) # Save loss for each epoch



    def evaluation(self, model, device, eval_loader, criterion, lat_wgts_tensor):

        model.eval() # Set the model in training mode
        test_loss = 0.0 # Initiate test loss
        use_amp = True # For automatic mixed precision

        for i, (inputs, targets) in enumerate(eval_loader):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):# Automatic mixed precision
                with torch.no_grad():
                    outputs = model(inputs)

            test_loss += criterion(outputs*lat_wgts_tensor, targets*lat_wgts_tensor).item() # Loss
        self.test_loss_list.append(test_loss / (i+1))  # Save the loss for each epoch
        
        
    def main(self):
        """
        Run the model
        """
        # Set fixed random number seed
        torch.manual_seed(42)

        #Loading datasets
        train_dataset = TrainDataset(variable=self.variable, model=self.clim_model, normalisation=self.normalisation)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
        
        val_members = train_dataset.val_members()
        val_dataset = ValidationDataset(variable=self.variable, model=self.clim_model, normalisation=self.normalisation, val_members_list=val_members)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, generator=torch.Generator(device='cuda'))

        eval_dataset = EvalDataset(variable=self.variable, model=self.clim_model, normalisation=self.normalisation)
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
        
        # Define model, optimizer, and loss function
        device = "cuda" if torch.cuda.is_available() else sys.exit("no GPU available") # To use GPU to run the model
        torch.set_default_device(device)
        self.model.to(device) # Run the model on GPU
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        criterion = nn.MSELoss().cuda()

        best_val_loss = np.inf # To compare validation loss and save the best model
        self.train_loss_list = []
        self.val_loss_list = []
        self.test_loss_list = []

        # Define saving paths
        if self.save_path is not None:
            model_save_path = self.save_path / 'model'
            model_save_path.mkdir(parents=True, exist_ok=True)
            plot_save_path = self.save_path / 'plots'
            plot_save_path.mkdir(parents=True, exist_ok=True)
            numpy_save_path = self.save_path / 'numpy'
            numpy_save_path.mkdir(parents=True, exist_ok=True)    
            
        # For loss computation on the globe
        lat = np.load('/home/globc/techer/Lat.npy')
        wgts = np.sqrt(np.cos(np.deg2rad(lat)))[np.newaxis,:,np.newaxis]
        wgts_tensor = torch.tensor(wgts, dtype=torch.float32, requires_grad=False)
        lat_wgts_tensor = wgts_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
        start_timer()
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1) # Set learning rate parameters

        for epoch in range(self.nb_epochs):
            
            # Make nvtx profile for the last epoch of the last fold
            if epoch==self.nb_epochs-1:
                torch.cuda.cudart().cudaProfilerStart()
                
            nvtx.range_push("Epoch " + str(epoch)) # Create a marker for each epoch
                
            # Training
            nvtx.range_push("Train") # Create a marker for training
            self.train(self.model, device, train_loader, optimizer, epoch, criterion, lat_wgts_tensor, model_save_path)  # Training
            nvtx.range_pop() # End of the marker for training
            np.save(numpy_save_path / 'train_loss', self.train_loss_list) # Save train loss
            np.savez(numpy_save_path / 'grad_weights', self.grad_norms) # Save gradients
            
            #Validation 
            nvtx.range_push("Validate") # Create a marker for validation step
            self.validate(self.model, device, val_loader, epoch, criterion, lat_wgts_tensor) # Validating      
            nvtx.range_pop() # End of the marker for validation step
            np.save(numpy_save_path / 'val_loss', self.val_loss_list) # Save validation loss
            
            scheduler.step()
            nvtx.range_pop() #  End of the marker for the epoch

             # Test for one epoch if the validations score is the best then save the best model
            if self.val_loss_list[-1] < best_val_loss:
                best_val_loss = self.val_loss_list[-1]
                leave = False
                if self.save_path is not None:
                    torch.save(self.model.state_dict(), model_save_path / 'best_model.pth')

        if self.save_path is not None and (model_save_path / 'best_model.pth').exists():
            print('\nLoad best model.')
            self.model.load_state_dict(torch.load(model_save_path / 'best_model.pth')) # Load the best model according to validation score
            
            # Evaluation
            self.evaluation(self.model, device, eval_loader, criterion, lat_wgts_tensor) # Evaluation of the model
            np.save(numpy_save_path / 'test_loss', self.test_loss_list) # Save evaluation/test loss


        print('\nScore test dataset:', self.test_loss_list[0])

        # Save model summary
        with open(self.save_path / 'model_summary.txt', 'w') as f:
            with redirect_stdout(f):
                #print('------- Model Summary')
                #print(summary(self.model, input_size=inputs.size())) 
                summary(self.model, input_size=(self.batch_size, 1, 122, 72, 144))
                print('\n------- Training setting:')
                print('    - number of nb_epochs:', self.nb_epochs)
                print('    - loss function:', criterion)
                print('    - optimizer:', optimizer)

            np.save(numpy_save_path / 'train_loss', self.train_loss_list)    
            np.save(numpy_save_path / 'val_loss', self.val_loss_list)
            np.save(numpy_save_path / 'test_loss', self.test_loss_list)

        end_timer_and_print("Default precision:")
        
        # Stop nvtx profiling for the last epoch of the last fold
        if epoch == self.nb_epochs-1:
            torch.cuda.cudart().cudaProfilerStop()
        
# Running the code        

save_path = Path("/scratch/globc/techer/ia/models/UNet_56") # Saving path of the trained model
dl_model = UNet_SiLU(in_channels=1, out_channels=1) # Define the deep learning model
model_dl = Model(variable='tas', 
                   clim_model='CESM2', 
                   model=dl_model, 
                   normalisation='norm3', 
                   nb_epochs=100, 
                   batch_size=8, 
                   save_path=save_path)
model_dl.main() # Run the deep learning model
