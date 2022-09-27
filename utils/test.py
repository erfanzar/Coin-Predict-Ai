import math


def dataloader_test(dataloader):
    data_iter = iter(dataloader)
    total = dataloader.dataset.__len__()
    iterations = math.ceil(total / dataloader.batch_size)
    print(f'Number of Iterations : {iterations}')
    print(f'Number of batch : {dataloader.batch_size}')
    for batch_index in range(iterations):
        x, y = data_iter.__next__()
    print("Working Fine [ * ]")
