import torch

def train(model, device, train_loader, optimizer, epoch, batch_size):

    # set model to training mode
    model.train()

    for batch_idx, data in enumerate(train_loader):
        
        # add input channel to data
        data = data.float().unsqueeze(1).to(device) # shape = (batch_size, audio_channel, audio_window) = (8, 1, 20480)

        # initialize gradient
        optimizer.zero_grad()

        # initialize hidden state
        hidden = torch.zeros((1, len(data), 256)).to(device)

        # forward
        #acc, loss, hidden = model(data, hidden)
        loss, hidden = model(data, hidden)

        # backward
        loss.backward()

        # update
        optimizer.step()

        # print
        if batch_idx % 50 == 0:
            trained = batch_idx * len(data)
            total = len(train_loader.dataset)
            percent = trained / total * 100
            #print(f"Train Epoch {epoch} [{trained} / {total} ({percent:.0f}%)]\tAccruacy : {acc:.4f}\tLoss : {loss.item():.6f}")
            print(f"Train Epoch {epoch} [{trained} / {total} ({percent:.0f}%)]\tLoss : {loss.item():.6f}")

        
