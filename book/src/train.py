import torch
from tqdm.autonotebook import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, opt, loss_fun, device=DEVICE):
    avg_loss = 0.0
    for i, (x, y) in (bar := tqdm(enumerate(loader))):
        # First check that there are valid samples
        if not len(x): continue
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        model.train()
        yhat = model(x)
        loss = loss_fun(yhat, y)
        loss.backward()
        opt.step()
        avg_loss += loss.cpu().detach().float().numpy()
        bar.set_description(f'Training loss: {loss:.2e}')
    bar.container.close()
    return avg_loss / i


def valid_epoch(model, loader, opt, loss_fun, device=DEVICE):    
    avg_loss = 0.0
    for i, (x, y) in (bar := tqdm(enumerate(loader))):
        # First check that there are valid samples
        if not len(x): continue
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        model.eval()
        with torch.no_grad():
            yhat = model(x)
        loss = loss_fun(yhat, y)
        avg_loss += loss.cpu().float().numpy()
        bar.set_description(f'Validation loss: {loss:.2e}')
    bar.container.close()
    return avg_loss / i


def train_model(model, loader, opt, loss_fun, max_epochs, device=DEVICE):
    train_loss = []
    valid_loss = []
    for e in (bar := tqdm(range(max_epochs))):
        vl = valid_epoch(model, valid_pipe, opt, loss_fun)
        tl = train_epoch(model, train_pipe, opt, loss_fun)
        train_loss.append(tl), valid_loss.append(vl)
        bar.set_description(f'Train loss: {tl:0.1e}, valid loss: {vl:0.1e}')