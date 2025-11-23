import torch
import torch.nn as nn
print(torch.__version__)
print("Hello")
torch.manual_seed(42)
input=[1,2,3,4,5]
kernel=[1,0,-1]
input=torch.tensor(input,dtype=torch.float32)

kernel=torch.tensor(kernel,dtype=torch.float32)


conv=nn.Conv1d(in_channels=1,out_channels=1,kernel_size=3,bias=False)


conv.weight.data=kernel.view(1,1,3)
input=input.view(1,1,5)

output=conv(input)


input_size=5
kernel_size=3
output_size=input_size-kernel_size+1
output2=torch.empty(output_size)

for i in range(output_size):
    s=0.0
    for j in range(kernel_size):
        s += input[0, 0, i + j] * kernel[j]
    output2[i]=s

print(output2)

print(output)




