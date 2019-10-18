from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Import TrainTrack Bot
from traintrack import TrainTrack

telegram_token = "934547307:AAEb1Pqhk2iXvPrs1pLfHCrZTPehSx0dIkU"  # bot's token
# user id is optional, however highly recommended as it limits the access to you alone.
telegram_user_id = 734383954  # telegram user id (integer):
# Create a TrainTrack Bot instance
TrainTrack = TrainTrack(token=telegram_token, user_id=telegram_user_id)
# Activate the bot
TrainTrack.activate_bot()


# Unmodified definition of network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# TrainTrack requires some modifications during training process
'''
Args:
    msg: The update message that prints the results of current epoch at the end of process
         Used to pass as a parameter to the update_message() function
'''
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item())
            print(msg)

            '''
            ~~~ Send update messages to the user ~~~
            With update_message() function you can send your training results to the user
            '''
            TrainTrack.update_message(msg)

            '''
            ~~~ Append train loss to TrainTrack ~~~
            Note that you can either append a floating point value or a 1d tensor
            Throughout the code you should preserve your decision and append the same type
            In conclusion either one of the following lines will work. Make sure to choose only one
            Similarly if you calculate and want to plot the training accuracy you can plot it using
                cumulate_train_acc(<accuracy>)
            '''
            TrainTrack.cumulate_train_loss(loss.item())
            # TrainTrack.cumulate_train_loss(loss)


# TrainTrack requires some modifications during test process
'''
Args:
    msg: The update message that prints the results of current epoch at the end of process
         Used to pass as a parameter to the update_message() function
'''
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    msg = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    print(msg)
    '''
    ~~~ Send update messages to the user ~~~
    With update_message() function you can send your test results to the user
    '''
    TrainTrack.update_message(msg)
    '''
    ~~~ Append test loss and accuracy to TrainTrack ~~~
    Note that you can either append a floating point value or a 1d tensor
    Throughout the code you should preserve your decision and append the same type
    '''
    TrainTrack.cumulate_test_loss(test_loss)
    TrainTrack.cumulate_test_acc(correct / len(test_loader.dataset))


# TrainTrack requires some modifications in the main method
'''
Args:
    epoch: index of the current epoch
    stop_train_flag: TrainTrack bool to check whether the user requested to stop
                     the training process
    lr: TrainTrack floating point value to change the learning rate (Initial value: None)
'''
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        # Update the epoch variable in TrainTrack in order to keep track of
        # the current epoch
        TrainTrack.update_epoch(epoch)
        # Force break epoch loop when the user stops training
        if TrainTrack.stop_train_flag:
            break
        # Manually control learning rate using TrainTrack
        if TrainTrack.lr is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = TrainTrack.lr

        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    # Exit conditions handling for TrainTrack
    # Notifies the user whether the training has terminated or finished after completing all epochs
    if TrainTrack.stop_train_flag:
        print("Training stopped by {}!".format(TrainTrack.name))
        TrainTrack.send_message("Training stopped by {}!".format(TrainTrack.name))
    else:
        print("Training complete. {} out!".format(TrainTrack.name))
        TrainTrack.send_message("Training complete. {} out!".format(TrainTrack.name))
    # Stop TrainTrack Bot instance at the end of training
    TrainTrack.stop_bot()


if __name__ == '__main__':
    main()
