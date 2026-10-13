import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def train_regression(
    model,
    x_train, y_train,
    x_test, y_test,
    epochs=100,
    lr=1e-3,
    batch_size=128,
    optimizer_type="adam",
    device=None,
    freeze_rbfkan=False,
    fine_tune_lr=1e-4,
    fine_tune_epochs=50,
    update_grid_every=20,  # controllo aggiornamento griglia
    weight_decay = 0.0,
    model_name = "Model"
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    criterion = nn.MSELoss()

    # Freeze RBF-KAN se richiesto
    if freeze_rbfkan and hasattr(model, 'rbf_kan'):
        for p in model.rbf_kan.parameters():
            p.requires_grad = False

    # Selezione ottimizzatore
    if optimizer_type.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            history_size=20,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-9
        )
        use_closure = True
        # Batch per LBFGS: tutto il dataset, oppure massimo 5000 campioni
        current_batch_size = min(x_train.shape[0], 5000)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        use_closure = False
        current_batch_size = batch_size if batch_size > 0 else x_train.shape[0]

    train_loss, test_loss = [], []

    pbar = tqdm(range(epochs), desc=f"Training ({optimizer_type.upper()})", ncols=100)
    
    for epoch in pbar:
        model.train()

        # Aggiornamento griglia RBF/KAN se previsto
        if hasattr(model, 'update_grid') and epoch % update_grid_every == 0:
            model.update_grid(x_train)

        if use_closure:
            # LBFGS full-batch (o sottocampione se troppo grande)
            idx = torch.randperm(x_train.shape[0])[:current_batch_size]
            def closure():
                optimizer.zero_grad()
                pred = model(x_train[idx])
                loss = criterion(pred.view(-1, 1), y_train[idx].view(-1, 1))
                loss.backward()
                return loss
            optimizer.step(closure)
        else:
            # Adam mini-batch
            idx = torch.randperm(x_train.shape[0])[:current_batch_size]
            optimizer.zero_grad()
            pred = model(x_train[idx])
            loss = criterion(pred.view(-1, 1), y_train[idx].view(-1, 1))
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            pred_train = model(x_train)
            pred_test = model(x_test)
            train_rmse = torch.sqrt(criterion(pred_train.view(-1, 1), y_train.view(-1, 1))).item()
            test_rmse = torch.sqrt(criterion(pred_test.view(-1, 1), y_test.view(-1, 1))).item()

        train_loss.append(train_rmse)
        test_loss.append(test_rmse)
        pbar.set_description(f"| [{model_name}] | train RMSE: {train_rmse:.4f} | test RMSE: {test_rmse:.4f} |")

    # Fine-tuning RBF-KAN se richiesto
    if freeze_rbfkan and fine_tune_epochs > 0 and hasattr(model, 'rbf_kan'):
        for p in model.rbf_kan.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=fine_tune_lr)
        pbar_ft = tqdm(range(fine_tune_epochs), desc="Fine-tuning RBF-KAN", ncols=100)
        for _ in pbar_ft:
            model.train()
            idx = torch.randperm(x_train.shape[0])[:current_batch_size]
            optimizer.zero_grad()
            pred = model(x_train[idx])
            loss = criterion(pred.view(-1, 1), y_train[idx].view(-1, 1))
            loss.backward()
            optimizer.step()
            # Eval
            model.eval()
            with torch.no_grad():
                train_rmse = torch.sqrt(criterion(model(x_train).view(-1, 1), y_train.view(-1, 1))).item()
                test_rmse = torch.sqrt(criterion(model(x_test).view(-1, 1), y_test.view(-1, 1))).item()
            train_loss.append(train_rmse)
            test_loss.append(test_rmse)
            pbar_ft.set_description(f"| FT train RMSE: {train_rmse:.4f} | FT test RMSE: {test_rmse:.4f} |")

    return train_loss, test_loss, model(x_train).detach().cpu()

def train_classification(
    model,
    train_loader,
    test_loader,
    epochs=100,
    lr=1e-3,
    optimizer_name="adam",
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer")

    train_losses = []
    test_losses = []

    pbar_epochs = tqdm(range(epochs), desc="Training Progress")

    for epoch in pbar_epochs:
        model.train()
        running_train_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if xb.dim() > 2: 
                xb = xb.view(xb.size(0), -1) 

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * xb.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                if xb.dim() > 2: 
                    xb = xb.view(xb.size(0), -1)
                
                logits = model(xb)
                loss = loss_fn(logits, yb)
                running_test_loss += loss.item() * xb.size(0)
        
        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        pbar_epochs.set_postfix({
            'Tr_Loss': f"{epoch_train_loss:.4f}", 
            'Te_Loss': f"{epoch_test_loss:.4f}"
        })

    return train_losses, test_losses