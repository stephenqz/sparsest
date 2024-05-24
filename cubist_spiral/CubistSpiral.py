import numpy as np
import torch
import math

def get_octagon_vertices(N=17, noise_std=0):
    theta = np.linspace(1*math.pi,5*math.pi,N) + 1.6*1/8*2*math.pi

    r = 10*theta + math.pi
    data = np.array([np.cos(theta)*r, np.sin(theta)*r]).T
    x = data + noise_std*np.random.randn(N,2)

    c = 15
    x_a = x + c*np.array([np.cos(theta), np.sin(theta)]).T
    x_b = x - c*np.array([np.cos(theta), np.sin(theta)]).T
    min_a = np.amin(x_a, axis=0)
    min_b = np.amin(x_b, axis=0)
    shift = np.amin(np.stack((min_a, min_b)), axis=0)
    return torch.from_numpy(x_a-shift), torch.from_numpy(x_b-shift)
    
def linspace(v, noise_std=0):
    dist = []
    for i in range(len(v)-1):
        cur = v[i,:]
        next = v[i+1,:]
        dist.append(torch.norm(cur-next))
    dist = torch.tensor(dist)
    prob = dist / torch.sum(dist)

    sample_size = 50000
    x = []
    for i in range(len(v)-1):
        cur = v[i,:].unsqueeze(0)
        next = v[i+1,:].unsqueeze(0)
        t = torch.linspace(0, 1, int(prob[i] * sample_size / 2)).unsqueeze(1)
        clean = t*cur + (1-t)*next
        x.append(clean + torch.randn_like(clean)*noise_std)
    return torch.cat(x)

def get_data():
    v1, v2 = get_octagon_vertices()
    x1 = linspace(v1)
    x2 = linspace(v2)
    x = torch.cat([x1, x2]).float()

    x = x - torch.mean(x, dim=0, keepdim=True)
    x = x / torch.std(x, dim=0, keepdim=True)
    y1 = torch.zeros(len(x1))
    y2 = torch.ones(len(x2))
    y = torch.cat([y1, y2]).unsqueeze(-1)

    return x, y
