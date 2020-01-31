from model import *
from sampler import *

batch_size = 64
num_points = 64
num_labels = 1

pointnet = PointNet(num_points, num_labels)
        
criterion = nn.BCELoss()
optimizer = optim.Adam(pointnet.parameters(), lr=0.001)

loss_list = []

for iteration in range(10000+1):
    
    pointnet.zero_grad()
    
    input_data, labels = data_sampler(batch_size, num_points)
    
    output = pointnet(input_data)
    output = nn.Sigmoid()(output)
    
    error = criterion(output, labels)
    error.backward()
    
    optimizer.step()

    loss_list.append(error.item())
    
    if iteration % 10 == 0:
        print('Iteration : {}   Loss : {}'.format(iteration, error.item()))