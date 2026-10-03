    print(f"Training examples: {len(train_loader.dataset):,}")
    print(f"Validation examples: {len(val_loader.dataset):,}")
    print(f"Test examples: {len(test_loader.dataset):,}")

    # CPU is fast enough for DLinear and keeps this run reproducible.
    device = torch.device("cpu")
    print(f"Using device: {device}")

    model = DLinear(
        lookback=LOOKBACK,
        horizon=HORIZON,
        kernel_size=KERNEL_SIZE,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("No model state was saved.")

    model.load_state_dict(best_state)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch.to(device))
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y_batch.numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    predictions_mw = predictions * train_std + train_mean
    actuals_mw = actuals * train_std + train_mean

    metrics = evaluate(actuals_mw, predictions_mw)

    print("\nDLinear Test Results")
    print(f"MAE:  {metrics['mae_mw']:.2f} MW")
    print(f"RMSE: {metrics['rmse_mw']:.2f} MW")
    print(f"WAPE: {metrics['wape_percent']:.4f}%")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame([
        {
            "model": "DLinear",
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "kernel_size": KERNEL_SIZE,
            "best_epoch": best_epoch,
            "MAE_MW": metrics["mae_mw"],
            "RMSE_MW": metrics["rmse_mw"],
            "WAPE_percent": metrics["wape_percent"],
        }
    ])

    result.to_csv(RESULTS_PATH, index=False)
    print(f"Saved metrics to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
