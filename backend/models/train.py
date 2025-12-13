import torch
import tqdm
from torch import nn
import torch.optim as optim
from backend.models.fretboard_cnn import *
from backend.models.extract_synthtab import *


# def create_data_loader(train_data, batch_size):
#     train_dataloader = DataLoader(train_data, batch_size=batch_size)
#     return train_dataloader

def train_one_epoch(model, data_loader, criterion, optimiser, device):
    model.train()
    running_loss = 0.0
    correct_joint = 0
    correct_str = 0
    correct_fret = 0
    total = 0
    for pitches, strings, frets in data_loader:
        # move to device
        pitches, strings, frets = pitches.to(device), strings.to(device), frets.to(device) 

        targets = (strings * 25) + frets
        
        # forward pass
        logits = model(pitches)
        
        # # calculate loss 
        # loss_string = crit_string(string_logits, strings)
        # loss_fret = crit_fret(fret_logits, frets)
        
        # # increasing importance of fret
        # loss = loss_string + (2 *loss_fret)
        
        loss = criterion(logits, targets)
        
        # backpropogation error and updating weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        running_loss += loss.item() * pitches.size(0)
        
        preds = torch.argmax(logits, dim=1)
        # fret_pred = torch.argmax(fret_logits, dim=1)
        correct_joint += (preds == targets).sum().item()

        total += pitches.size(0)
    
    epoch_loss = running_loss / total
    epoch_string_accuracy = correct_str / total
    epoch_fret_accuracy = correct_fret / total
    
    # return epoch_loss, epoch_string_accuracy, epoch_fret_accuracy 
    return (running_loss / total, 
            correct_joint / total, 
            correct_str / total, 
            correct_fret / total)
        
def validate(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct_joint = 0
    correct_str = 0
    correct_fret = 0
    total = 0
    
    with torch.no_grad():
        for pitches, strings, frets in tqdm.tqdm(data_loader, desc="Training", leave=False):
            pitches = pitches.to(device)
            strings = strings.to(device)
            frets = frets.to(device)
            
            targets = (strings * 25) + frets
            
            # forward pass
            logits = model(pitches)
            
            # loss
            # loss_string = crit_string(string_logits, strings)
            # loss_fret = crit_fret(fret_logits, frets)
            # loss = loss_string + loss_fret
            loss = criterion(logits, targets)
            
            running_loss += loss.item() * pitches.size(0)
            
            
            preds = torch.argmax(logits, dim=1)
            correct_joint += (preds == targets).sum().item()
            # Accuracy
         
            pred_strings = preds // 25
            pred_frets = preds % 25
            
            correct_str += (pred_strings == strings).sum().item()
            correct_fret += (pred_frets == frets).sum().item()
            
            total += pitches.size(0)
            
    epoch_loss = running_loss / total
    epoch_string_accuracy = correct_str / total
    epoch_fret_accuracy = correct_fret / total
    
    return (running_loss / total, 
            correct_joint / total, 
            correct_str / total, 
            correct_fret / total)
    
def train(model, data_loader, val_loader, criterion, optimiser, scheduler, device, epochs):
    """
    Trains the fretboard CNN, saving the best performing model 
    """
    # settinng best val to infinity
    best_val_loss = float('inf')
    
    for i in range(epochs):
        t_loss, t_joint, t_str, t_fret = train_one_epoch(
            model, data_loader, criterion, optimiser, device
        )
        v_loss, v_joint, v_str, v_fret = validate(
            model, val_loader, criterion, device
        )
        
        if scheduler:
            scheduler.step(v_loss)
        
        print(f'\nEpoch {i+1}/{epochs}:')
        print(f'  Train | Loss: {t_loss:.4f} | Joint Acc: {t_joint:.1%} (Str: {t_str:.1%}, Fret: {t_fret:.1%})')
        print(f'  Val   | Loss: {v_loss:.4f} | Joint Acc: {v_joint:.1%} (Str: {v_str:.1%}, Fret: {v_fret:.1%})')
        
        # Save best performing model
        if v_loss < best_val_loss:
            os.makedirs('models', exist_ok=True)
            best_val_loss = v_loss
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'val_loss': v_loss,
                'val_joint_acc': v_joint,
            }, 'backend/models/models/best_fretboard_cnn.pt')
            print(f'  -> Saved best model (val_loss: {v_loss:.4f})')

    print("Training complete.")
    return model

def train_model(file_dir, epochs, batch_size, context_window=5, max_files=10, learning_rate=0.001):
    # www.datacamp.com/tutorial/pytorch-cnn-tutorial
    # Valerio Velardo
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # loading data
    data_loader, val_loader = create_data_loader(
        file_dir=file_dir, 
        batch_size=batch_size,
        train_split=0.8,
        context_window=context_window,
        max_files=max_files
    )
    
    # creating model
    model = FretBoardCNN(context_window, 32)
    model = model.to(device)
    
    # criterion loss function
    criterion = nn.CrossEntropyLoss()
    
    # optimiser
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    
    # lowers learning rate if validation does not improve anymore
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=3, factor=0.5, verbose=True)
    
    train(model, data_loader, val_loader, criterion, optimiser, scheduler, device, epochs)
    
    return model
    
if __name__ == '__main__':
    model = train_model(
        file_dir=os.path.abspath('backend/resources/all_jams_midi_V2_60000_tracks/outall'),
        epochs=5,
        batch_size=1024,
        context_window=5,    
        max_files=5000,  
        learning_rate=0.001)
    