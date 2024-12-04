import torch
import torch.nn.functional as F
import numpy as np

from layers import Linear, ReLu, Softmax, MSELoss

def test_linear():
    np.random.seed(54)
    batch_size = 2
    in_dim = 8
    out_dim = 4

    #generate weights and biases
    W = np.random.randn(in_dim, out_dim).astype(np.float32)
    biases = np.random.randn(out_dim).astype(np.float32)

    #initialize my linear layer
    my_layer = Linear(in_dim, out_dim)
    my_layer.W = W
    my_layer.bias = biases

    #initialize my input and grad_out
    my_input = np.random.randn(batch_size, in_dim).astype(np.float32)
    out_grad = np.random.randn(batch_size, out_dim).astype(np.float32)

    #my forward
    my_output = my_layer.forward(my_input)

    #my backward
    my_layer.backward(out_grad)



    #initialize torch linear layer
    torch_layer = torch.nn.Linear(in_dim, out_dim, bias=True)
    torch_layer.weight.data = torch.tensor(W.T)
    torch_layer.bias.data = torch.tensor(biases)    

    #initialize torch input
    torch_input = torch.tensor(my_input, requires_grad=True)
    torch_out_grad = torch.tensor(out_grad)

    #torch forward and backward
    torch_output = torch_layer(torch_input)
    torch_output.backward(torch_out_grad)

    assert np.allclose(my_output, torch_output.detach().numpy(), atol=1e-8), \
            "Forward outputs don't match"
    
    assert np.allclose(my_layer.dW, torch_layer.weight.grad.numpy().T, atol=1e-8), \
            "Weights gradients don't match"
    
    assert np.allclose(my_layer.db, torch_layer.bias.grad.numpy(), atol=1e-8), \
            "Bias gradients do not match!"

    assert np.allclose(my_layer.dX, torch_input.grad.numpy(), atol=1e-8), \
            "Input gradients do not match!"

    print("Linear layer test passed!")




