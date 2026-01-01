import os
os.environ['USE_GPU']='1'
import tqdm
from src.utils.cupy_numpy import np
from src.optimizer.SDLM import SDLM
from model import LeNet5
from src.utils.data_loader import load_data


if hasattr(np, 'cuda'):  # CuPy
    print(f"✅ Using GPU: {np.cuda.Device(0).compute_capability}")
    # Test GPU
    test = np.array([1, 2, 3])
    print(f"Test array device: {test.device if hasattr(test, 'device') else 'CPU'}")
else:
    print("❌ Using CPU (NumPy)")


def train(model, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size):
    """
        Training based on LeNet-5 Architecture 
        1. Load the data
        2. Split into training and validation set
        3. Forward Pass:  Train the model in the batch of 128 
        4. Calculate the loss for the batch
        5. Backward Pass: Backpropagate and calculate the gradient using backward() function
        6. Optimization: Update the training parameters using the optimizer SDLM
    """

    #Set the learning rate based on layer as per the LeNet-5 paper
    lr_schedule = {
        'C1.W': 0.001,   # input → C1
        'C1.b': 0.002,   # bias = 2× weight LR
        'S2.alpha': 0.005,
        'S2.beta': 0.005,
        'C3.W0': 0.005, 'C3.W1': 0.005,'C3.W2': 0.005, 'C3.W3': 0.005,'C3.W4': 0.005,'C3.W5': 0.005,'C3.W6': 0.005,'C3.W7': 0.005,'C3.W8': 0.005,'C3.W9': 0.005,'C3.W10': 0.005,'C3.W11': 0.005,'C3.W12': 0.005,'C3.W13': 0.005,'C3.W14': 0.005,'C3.W15': 0.005,
        'C3.b': 0.01,
        'S4.alpha': 0.01,
        'S4.beta': 0.01,
        'C5.W': 0.01,
        'C5.b': 0.02,
        'F6.W': 0.01,
        'F6.b': 0.02,
    }

    for name, lr in lr_schedule.items():
        optimizer.set_learning_rate(name, lr)
    
 

    #Training through epochs
    n_samples = int(X_train.shape[0])  # Ensure Python int
    n_batches = n_samples // batch_size
    history = {
        'train_loss' : [],
        'train_acc' : [],
        'val_loss' : [],
        'val_acc' : [],
        'learning_rate' : [],
    }

    # Helper function to convert CuPy arrays to Python scalars
    def to_scalar(x):
        """Convert CuPy/NumPy array to Python scalar"""
        if hasattr(x, 'item'):
            return x.item()
        elif hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
            return float(x)
        return float(x)
    
    for epoch in range(epochs):
        # Generate permutation - ensure it's on the same device as data
        indices = np.random.permutation(n_samples)
        
        epoch_loss = 0.0
        epoch_correct = 0

        pbar = tqdm.tqdm(range(0, n_samples, batch_size), desc = f"Epoch {epoch + 1} / {epochs}", leave = False)
        # batches = list(range(0, n_samples, batch_size))

        for start in pbar:
        # for idx, start in enumerate(batches):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start: end]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            #Forward Pass
            Y_batch = model.forward(X_batch)
            
            #Compute Loss - convert to Python scalar
            loss = model.compute_loss(Y_batch, y_batch)
            loss_scalar = to_scalar(loss)
            epoch_loss += loss_scalar

            #Backward Pass
            dout = model.MAP_Loss.backward()
            model.backward(dout)
            
            #Optimization
            named_params, grads = model.get_named_params_and_grads()
            optimizer.step(named_params, grads)

            #Accuracy - convert to Python scalar
            predictions = np.argmin(Y_batch, axis = 1)
            correct = np.sum(predictions == y_batch)
            epoch_correct += to_scalar(correct)
            
            # Update progress bar with scalar value
            pbar.set_postfix({"Loss" : f"{loss_scalar: .4f}"})
            # if idx % 50 == 0:
                # print(f"Batch {idx + 1} | Loss: {loss:.4f}")
        
        #Compute Epoch Metrics
        train_loss = epoch_loss / n_batches
        train_acc = epoch_correct / n_samples 
        
        #For validation dataset
        val_pred = model.forward(X_val)
        val_loss = model.compute_loss(val_pred, y_val)
        val_acc = model.get_accuracy(X_val, y_val)
        
        # Convert to Python scalars for printing
        val_loss_scalar = to_scalar(val_loss)
        val_acc_scalar = to_scalar(val_acc)

        print(f"Epoch {epoch + 1:2d} | Training Loss: {train_loss:.4f} | Training_Acc: {train_acc:.4f} | Val_Acc: {val_acc_scalar:.4f} | Val_Loss: {val_loss_scalar:.4f}")

        history['train_acc'] = train_acc
        history['train_loss'] = train_loss
        history['val_loss'] = val_loss_scalar
        history['val_acc'] = val_acc_scalar
    
    return model, history

def save_model(model: LeNet5, filepath = 'checkpoints/LeNet5_weights.npz'):

    named_params, _  = model.get_named_params_and_grads()

    params_to_save = {}
    for name, param in named_params:
        params_to_save[name] = param
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if "/" in filepath else None
    np.savez(filepath, **params_to_save)

    print ("-" * 20)
    print(f"Model saved to {filepath}")
    print ("-" * 20)

def load_model(model: LeNet5, filepath = 'checkpoints/LeNet5_weights.npz'):
    """
    Load the saved model
    """

    named_params, _ = model.get_named_params_and_grads()
    
    data = np.load(filepath)

    for name, param in named_params:

        if name not in data:
            raise KeyError(f"Parameter {name} not found in the saved model")
        
        saved_params = data[name]

        if param.shape != saved_params.shape:
            raise ValueError(
                f"Shape mismatch for {name}"
                f"Model {param.shape} != Saved {saved_params.shape}"
            )     

        param[...] = saved_params
    
    print(f"Model parameters laoded from {filepath}")


if __name__ == "__main__":

    if os.environ.get("USE_GPU") == "1":
        try:
            import cupy as cp
            print(f"GPU Device: {cp.cuda.Device(0).compute_capability}")
            print(f"GPU Memory: {cp.get_default_memory_pool().get_limit() / 1024**3:.2f} GB")
        except:
            print("Warning: GPU enabled but CuPy not working properly")

        
    model = LeNet5()
    optimizer = SDLM()

    epochs = 20
    batch_size = 128
    
    #Load the data
    print("Loading MNIST dataset....")
    X_train, X_test, y_train, y_test = load_data() #This will give images of shapes (batch, channel, height, width) = (batch, 1, 32, 32)

    #Splitting the training data to training and validation set
    X_val = X_train[:5000]
    y_val = y_train[:5000]

    X_train = X_train[5000: ]
    y_train = y_train[5000: ]

    #Fit the model and train
    history = train(model, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size)

    #Save the model
    save_model(model)
    
    # Helper function to convert CuPy arrays to Python scalars
    def to_scalar(x):
        """Convert CuPy/NumPy array to Python scalar"""
        if hasattr(x, 'item'):
            return x.item()
        elif hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
            return float(x)
        return float(x)
    
    test_pred = model.forward(X_test)
    test_loss = model.compute_loss(test_pred, y_test)
    test_acc = model.get_accuracy(X_test, y_test)
    test_acc_scalar = to_scalar(test_acc)
    test_loss_scalar = to_scalar(test_loss)
    print(f"Test Accuracy: {test_acc_scalar * 100:.2f}%")
    print(f"Test Prediction: {test_loss_scalar:.4f}")

